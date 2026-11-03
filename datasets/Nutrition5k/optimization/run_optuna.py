import json
import optuna
import os
import glob


# ================= CONFIGURATION =================
# 1. PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "../data", "ground truth", "train_set_optuna.json")
PREDICTIONS_BASE_DIR = os.path.join(BASE_DIR, "../data", "optuna")

# 2. OPTUNA STRATEGY
# Set the strategy to use here, or via the OPTUNA_STRATEGY environment variable.
# Available strategies (see STRATEGY_CATALOG):
#   "linear_all5"      -> 5 independent scales  [default, already run]
#   "linear_mass_only" -> optimizes only the mass (1 parameter)
#   "power_law_all5"   -> power law (scale * pred ^ power) (10 parameters)
#   "piecewise_all5"   -> threshold-based correction for small/large values (15 parameters)
#   "linear_bias_all5" -> linear model with scale and bias (10 parameters)
OPTUNA_STRATEGY = os.environ.get("OPTUNA_STRATEGY", "all")

# Strategy catalog
STRATEGY_CATALOG = {
    "linear_all5": {
        "nutrients": ["mass", "cal", "fat", "carbs", "protein"],
        "description": "Independent linear multiplier for each of the 5 macronutrients",
        "param_type": "linear_scale",
        "n_trials": 5000,
    },
    "linear_mass_only": {
        "nutrients": ["mass"],
        "description": "Linear multiplier optimized for total mass only",
        "param_type": "linear_scale",
        "n_trials": 5000,
    },
    "power_law_all5": {
        "nutrients": ["mass", "cal", "fat", "carbs", "protein"],
        "description": "Power-law correction: scale * (pred_val ^ power) for each nutrient (10 parameters)",
        "param_type": "power_scale",
        "n_trials": 5000,
    },
    "piecewise_all5": {
        "nutrients": ["mass", "cal", "fat", "carbs", "protein"],
        "description": "Threshold multiplier: scale_small if pred < threshold, otherwise scale_large (15 parameters)",
        "param_type": "piecewise_scale",
        "n_trials": 5000,
    },
    "linear_bias_all5": {
        "nutrients": ["mass", "cal", "fat", "carbs", "protein"],
        "description": "Linear model with scale and bias (adj_val = pred * scale + bias)",
        "param_type": "linear_scale_bias",
        "n_trials": 5000,
    },
}


# ==================================================

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# Load the ground truth once at startup
print("Loading ground truth...")
ground_truth_data = load_json(GROUND_TRUTH_FILE)
gt_dict = {item['food_id']: item for item in ground_truth_data}


def get_gt_val(gt, nut):
    """Extracts the ground truth value for a given nutrient."""
    gt_nut = gt.get("total_macronutrients", {})
    if nut == "mass":
        return gt_nut.get("total_mass_grams", gt.get("total_mass", 0))
    elif nut == "cal":
        return gt_nut.get("calories", gt.get("total_calories", 0))
    elif nut == "fat":
        return gt_nut.get("fat", gt.get("total_fat", 0))
    elif nut == "carbs":
        return gt_nut.get("carbohydrates", gt.get("total_carbs", 0))
    elif nut == "protein":
        return gt_nut.get("protein", gt.get("total_protein", 0))
    return 0


def get_pred_val(pred, nut):
    """Extracts the predicted value for a given nutrient."""
    pred_block = pred.get("predicted_nutrition", pred.get("predicted_nutrition_step2", {}))
    pred_nut_dict = pred_block.get("total_macronutrients", {})
    if nut == "mass":
        ingredients = pred_block.get("visible_ingredients", []) + pred_block.get("hidden_ingredients", [])
        return sum(ing.get("weight_grams", 0) for ing in ingredients)
    elif nut == "cal":
        return pred_nut_dict.get("calories", 0)
    elif nut == "fat":
        return pred_nut_dict.get("fat", 0)
    elif nut == "carbs":
        return pred_nut_dict.get("carbohydrates", 0)
    elif nut == "protein":
        return pred_nut_dict.get("protein", 0)
    return 0


def optimize_linear_independent(predictions_data, strategy_cfg):
    """Strategy: independent linear scale for each nutrient."""
    nutrients = strategy_cfg["nutrients"]
    n_trials = strategy_cfg["n_trials"]
    best_params = {}
    total_mae_perc = 0.0

    for nut in nutrients:
        def objective(trial, _nut=nut):
            scale = trial.suggest_float(f"scale_{_nut}", 0.005, 5.0)
            errors = []
            gt_total = 0.0
            for pred in predictions_data:
                food_id = pred.get("food_id")
                if food_id not in gt_dict:
                    continue
                gt = gt_dict[food_id]
                gt_val = get_gt_val(gt, _nut)
                if gt_val <= 0 and _nut == "mass":
                    continue
                gt_total += gt_val
                pred_val = get_pred_val(pred, _nut)
                adj_val = max(0.0, pred_val * scale)
                errors.append(abs(adj_val - gt_val))
            if gt_total == 0 or len(errors) == 0:
                return 9999.0
            return (sum(errors) / gt_total) * 100

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        best_params.update(study.best_params)
        total_mae_perc += study.best_value

    return total_mae_perc / len(nutrients), best_params


def optimize_power_law_all5(predictions_data, strategy_cfg):
    """Strategy: power law. adj_val = scale * (pred_val ** power)"""
    nutrients = strategy_cfg["nutrients"]
    n_trials = strategy_cfg["n_trials"]
    best_params = {}
    total_mae_perc = 0.0

    for nut in nutrients:
        def objective(trial, _nut=nut):
            scale = trial.suggest_float(f"scale_{_nut}", 0.05, 5.0)
            power = trial.suggest_float(f"power_{_nut}", 0.3, 2.5)  # Wide range for the exponent
            errors = []
            gt_total = 0.0
            for pred in predictions_data:
                food_id = pred.get("food_id")
                if food_id not in gt_dict:
                    continue
                gt = gt_dict[food_id]
                gt_val = get_gt_val(gt, _nut)
                if gt_val <= 0 and _nut == "mass":
                    continue
                gt_total += gt_val
                pred_val = get_pred_val(pred, _nut)

                # Apply the non-linear formula (avoids a crash if pred_val is exactly 0)
                adj_val = scale * (max(0.0, pred_val) ** power)

                errors.append(abs(adj_val - gt_val))
            if gt_total == 0 or len(errors) == 0:
                return 9999.0
            return (sum(errors) / gt_total) * 100

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        best_params.update(study.best_params)
        total_mae_perc += study.best_value

    return total_mae_perc / len(nutrients), best_params


def get_pred_range(predictions_data, nut, margin=0.1):
    """
    Computes the [min, max] range of predicted values for a given nutrient
    in the train set, with a percentage safety margin.
    Used to dynamically set the bounds of the threshold parameter
    in optimize_piecewise_all5, instead of a fixed [1, 1500] range shared by all nutrients.
    """
    vals = [get_pred_val(pred, nut) for pred in predictions_data]
    vals = [v for v in vals if v > 0]  # discard zero/invalid values

    if not vals:
        # safe fallback if there are no valid values
        return 1.0, 1500.0

    v_min, v_max = min(vals), max(vals)

    # add a margin so the optimal threshold is not cut off
    # if it lies near the observed extremes
    range_span = v_max - v_min
    low = max(0.01, v_min - margin * range_span)
    high = v_max + margin * range_span

    return low, high


def optimize_piecewise_all5(predictions_data, strategy_cfg):
    """Strategy: Piecewise. Different scale depending on whether pred_val is below or above a threshold."""
    nutrients = strategy_cfg["nutrients"]
    n_trials = strategy_cfg["n_trials"]
    best_params = {}
    total_mae_perc = 0.0

    for nut in nutrients:
        # Threshold range specific to this nutrient
        threshold_low, threshold_high = get_pred_range(predictions_data, nut)

        def objective(trial, _nut=nut, _t_low=threshold_low, _t_high=threshold_high):
            threshold = trial.suggest_float(f"threshold_{_nut}", _t_low, _t_high)
            scale_small = trial.suggest_float(f"scale_small_{_nut}", 0.005, 5.0)
            scale_large = trial.suggest_float(f"scale_large_{_nut}", 0.005, 5.0)
            errors = []
            gt_total = 0.0

            for pred in predictions_data:
                food_id = pred.get("food_id")
                if food_id not in gt_dict:
                    continue
                gt = gt_dict[food_id]
                gt_val = get_gt_val(gt, _nut)
                if gt_val <= 0 and _nut == "mass":
                    continue
                gt_total += gt_val
                pred_val = get_pred_val(pred, _nut)

                if pred_val < threshold:
                    adj_val = max(0.0, pred_val * scale_small)
                else:
                    adj_val = max(0.0, pred_val * scale_large)

                errors.append(abs(adj_val - gt_val))
            if gt_total == 0 or len(errors) == 0:
                return 9999.0
            return (sum(errors) / gt_total) * 100

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        best_params.update(study.best_params)
        total_mae_perc += study.best_value

    return total_mae_perc / len(nutrients), best_params


def optimize_linear_bias_all5(predictions_data, strategy_cfg):
    """Strategy: linear model with bias. adj_val = pred_val * scale + bias"""
    nutrients = strategy_cfg["nutrients"]
    n_trials = strategy_cfg["n_trials"]
    best_params = {}
    total_mae_perc = 0.0

    for nut in nutrients:
        def objective(trial, _nut=nut):
            scale = trial.suggest_float(f"scale_{_nut}", 0.005, 5.0)

            # Bias range depends on the nutrient
            if _nut in ["mass", "cal"]:
                bias_limit = 500.0  # Wide range for total grams and calories
            else:
                bias_limit = 150.0  # Narrower range for fat, carbs, protein

            bias = trial.suggest_float(f"bias_{_nut}", -bias_limit, bias_limit)

            errors = []
            gt_total = 0.0

            for pred in predictions_data:
                food_id = pred.get("food_id")
                if food_id not in gt_dict:
                    continue

                gt = gt_dict[food_id]
                gt_val = get_gt_val(gt, _nut)
                if gt_val <= 0 and _nut == "mass":
                    continue

                gt_total += gt_val
                pred_val = get_pred_val(pred, _nut)

                # Apply the linear transformation (mx + q)
                adj_val = max(0.0, pred_val * scale + bias)

                errors.append(abs(adj_val - gt_val))

            if gt_total == 0 or len(errors) == 0:
                return 9999.0

            return (sum(errors) / gt_total) * 100

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        best_params.update(study.best_params)
        total_mae_perc += study.best_value

    return total_mae_perc / len(nutrients), best_params


# Strategy -> optimization function map
STRATEGY_FUNCTIONS = {
    "linear_all5": optimize_linear_independent,
    "linear_mass_only": optimize_linear_independent,
    "power_law_all5": optimize_power_law_all5,
    "piecewise_all5": optimize_piecewise_all5,
    "linear_bias_all5": optimize_linear_bias_all5,
}


def optimize_for_file(predictions_file, strategy_name):
    """Entry point: optimizes a predictions file with the given strategy."""
    if strategy_name not in STRATEGY_CATALOG:
        raise ValueError(f"Strategy '{strategy_name}' not found. "
                         f"Available: {list(STRATEGY_CATALOG.keys())}")
    strategy_cfg = STRATEGY_CATALOG[strategy_name]
    opt_fn = STRATEGY_FUNCTIONS[strategy_name]
    predictions_data = load_json(predictions_file)
    print(f"   Strategy: {strategy_name} | Nutrients: {strategy_cfg['nutrients']} | "
          f"Trials: {strategy_cfg['n_trials']}")
    return opt_fn(predictions_data, strategy_cfg)


def run_strategy(strategy_name, json_files):
    """
    Runs a full strategy on all json_files and saves the summary.
    Each Optuna study uses n_jobs=-1 to exploit all available cores.
    Skips the optimization if the results are already present in the summary.
    """
    strategy_cfg = STRATEGY_CATALOG[strategy_name]
    folder_results = {}

    prefix = f"[{strategy_name}]"
    print(f"\n{prefix} ▶ Start | {strategy_cfg['description']} | "
          f"Nutrients: {strategy_cfg['nutrients']} | Trials: {strategy_cfg['n_trials']}")

    for filepath in sorted(json_files):
        config_name = os.path.basename(filepath)
        folder_path = os.path.dirname(filepath)
        strategy_folder = os.path.join(folder_path, strategy_name)
        summary_file = os.path.join(strategy_folder, "optuna_summary.json")

        # Skip files already present in the summary
        already_done = False
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

                # If the current file was already computed earlier, skip it
                if config_name in existing_data:
                    already_done = True
                    print(f"{prefix} -> Optimization skipped (already computed): {config_name}")

                    # Keep the existing entry so it is not lost when saving at the end
                    if strategy_folder not in folder_results:
                        folder_results[strategy_folder] = {}
                    folder_results[strategy_folder][config_name] = existing_data[config_name]
            except Exception:
                pass

        if already_done:
            continue

        print(f"{prefix} -> Optimizing: {config_name}")
        try:
            best_mae, best_params = optimize_for_file(filepath, strategy_name)
            print(f"{prefix}    Mean MAE%: {best_mae:.2f}%")
        except Exception as e:
            print(f"{prefix} [!] Error on {filepath}: {e}")
            continue

        os.makedirs(strategy_folder, exist_ok=True)

        if strategy_folder not in folder_results:
            folder_results[strategy_folder] = {}
        folder_results[strategy_folder][config_name] = {
            "best_combined_mae_perc": round(best_mae, 2),
            "optimal_scalars": best_params
        }

    # Save JSON with _strategy_info at the top
    strategy_info_block = {
        "name": strategy_name,
        "description": strategy_cfg["description"],
        "nutrients_optimized": strategy_cfg["nutrients"],
        "param_type": strategy_cfg["param_type"],
        "n_trials": strategy_cfg["n_trials"],
    }

    for strategy_folder, results in folder_results.items():
        json_filename = os.path.join(strategy_folder, "optuna_summary.json")

        # Merge new data with old data if the summary already exists
        final_output = {"_strategy_info": strategy_info_block}
        if os.path.exists(json_filename):
            try:
                with open(json_filename, 'r', encoding='utf-8') as f:
                    old_data = json.load(f)
                    final_output.update(old_data)
            except Exception:
                pass

        # Overwrite/update the keys processed in this run
        final_output.update(results)

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4)
        print(f"[{strategy_name}] ✅ Saved: {json_filename}")

    return strategy_name, folder_results

if __name__ == "__main__":
    # RUN_MODE controls which folders to process:
    #   "both"                  (default) -> Semantic Matching + non-Semantic Matching
    #   "semantic_matching"               -> only *_semantic_matching folders
    #   "non_semantic_matching"           -> only folders without the _semantic_matching suffix
    RUN_MODE = os.environ.get("RUN_MODE", "both").lower()
    if RUN_MODE not in ("both", "semantic_matching", "non_semantic_matching"):
        print(f"[ERROR] RUN_MODE='{RUN_MODE}' is not valid. Choose among: both, semantic_matching, non_semantic_matching")
        exit(1)

    # Allows running a single strategy via env var.
    OPTUNA_STRATEGY_ENV = os.environ.get("OPTUNA_STRATEGY", "all")

    # Search all top-level folders (e.g. few_shot_standard, few_shot_yolo);
    # the extra "*" traverses the top-level folder layer
    pattern_semantic_matching     = os.path.join(PREDICTIONS_BASE_DIR, "*", "*_semantic_matching", "*.json")
    pattern_non_semantic_matching = os.path.join(PREDICTIONS_BASE_DIR, "*", "*", "*.json")   # matches everything

    if RUN_MODE == "semantic_matching":
        all_json_files = glob.glob(pattern_semantic_matching)
        print("Mode: Semantic Matching folders only (*_semantic_matching)")
    elif RUN_MODE == "non_semantic_matching":
        # Take all folders, then exclude those ending with _semantic_matching
        all_json_files = [
            f for f in glob.glob(pattern_non_semantic_matching)
            if not os.path.basename(os.path.dirname(f)).endswith("_semantic_matching")
        ]
        print("Mode: non-Semantic Matching folders only")
    else:  # both
        all_json_files = glob.glob(pattern_non_semantic_matching)
        print("Mode: all folders (Semantic Matching + non-Semantic Matching)")

    # Exclude: optuna_summary.json and files inside strategy subfolders
    strategy_names_set = set(STRATEGY_CATALOG.keys())
    json_files = [
        f for f in all_json_files
        if os.path.basename(f) != "optuna_summary.json"
        and not any(part in strategy_names_set for part in f.replace(os.sep, '/').split('/'))
    ]

    if not json_files:
        print("No JSON files found. Check the path or the search pattern.")
        exit(0)

    # Select strategies to run
    if OPTUNA_STRATEGY_ENV == "all" or OPTUNA_STRATEGY_ENV not in STRATEGY_CATALOG:
        strategies_to_run = list(STRATEGY_CATALOG.keys())
    else:
        strategies_to_run = [OPTUNA_STRATEGY_ENV]

    n_cores = os.cpu_count()
    print(f"\n{'=' * 60}")
    print(f"Strategies to run ({len(strategies_to_run)}): {strategies_to_run}")
    print(f"Prediction files found: {len(json_files)}")
    print(f"Available cores: {n_cores} (used by each Optuna study via n_jobs=-1)")
    print(f"{'=' * 60}\n")

    # Run strategies sequentially: each study uses all cores
    all_results = []
    for strategy_name in strategies_to_run:
        result = run_strategy(strategy_name, json_files)
        all_results.append(result)

    print(f"\n{'=' * 60}")
    print("✅ Batch optimization completed for all strategies!")
    for strategy_name, _ in all_results:
        print(f"   • {strategy_name}")
    print(f"{'=' * 60}")
