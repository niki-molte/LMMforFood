"""
runners.py
==========
Evaluation orchestration: reads the predictions, computes the metrics
(via Nutrition5kEvaluator), manages the CSV/JSON cache and calls the
charts in plots.py. Contains ONLY the pipelines that feed few-shot charts
(no per-model bar chart evaluation).
"""
import os
import csv
import json

from . import style as st
from .evaluator import Nutrition5kEvaluator
from .data_utils import parse_model_name
from . import plots

import re

import os
import json

import os
import json


def _load_optuna_scalars_for_model(folder, filename, strategy):
    """Load per-model Optuna scalars and match them to a prediction file."""
    is_yolo = "yolo" in filename.lower()
    technique_dir = "few_shot_yolo" if is_yolo else "few_shot_standard"

    summary_path = os.path.join("data", "optuna", technique_dir, folder,
                                strategy, "optuna_summary.json")

    if not os.path.exists(summary_path):
        print(f"[DEBUG] File missing on disk: {summary_path}")
        return None, summary_path

    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[DEBUG] JSON read error in {summary_path}: {e}")
        return None, summary_path

        # Clean extraction (strip everything to get the bare model name)
    core_model_name = filename.replace('test_predictions_nutrition5k_', '')
    core_model_name = core_model_name.replace('.json', '')
    core_model_name = core_model_name.replace('yolo_', '')
    clean_model_name = core_model_name.replace('_tb', '')

    if is_yolo:
           # For YOLO, the JSON has the _tb suffix
        lookup_key = f"optuna_train_predictions_{clean_model_name}_tb.json"
        fallback_key = f"optuna_train_predictions_{clean_model_name}.json"
    else:
        # For Standard, the JSON usually does NOT have the _tb suffix
        lookup_key = f"optuna_train_standard_predictions_{clean_model_name}.json"
        fallback_key = f"optuna_train_standard_predictions_{clean_model_name}_tb.json"

    # Primary key lookup
    if lookup_key in data:
        entry = data[lookup_key]
        if isinstance(entry, dict) and 'optimal_scalars' in entry:
            return entry['optimal_scalars'], summary_path
        else:
            print(f"[DEBUG] Found key {lookup_key} but 'optimal_scalars' is missing")

    # Fallback key lookup
    elif fallback_key in data:
        entry = data[fallback_key]
        if isinstance(entry, dict) and 'optimal_scalars' in entry:
            return entry['optimal_scalars'], summary_path
        else:
            print(f"[DEBUG] Found fallback key {fallback_key} but 'optimal_scalars' is missing")

    # No match found
    else:
        print(f"[DEBUG] Key not found. Looked for '{lookup_key}' or '{fallback_key}'. "
              f"Keys available in the JSON: {list(data.keys())}")

    return None, summary_path


# ---------------------------------------------------------------------------
# 1) Few-shot learning (NLP + Macro), 1/5/10 examples
# ---------------------------------------------------------------------------
def evaluate_few_shot_learning(resources, threshold,
                                few_shot_variants, base_dir="data/prediction",
                                folders=None, folder_to_prefix=None, output_dir=None,
                                csv_suffix="", run_label="1, 5, 10 Examples",
                                optuna_strategy="linear_all5"):
    if folders is None:
        folders = ["one_example", "five_example", "ten_example"]
    if folder_to_prefix is None:
        folder_to_prefix = {"one_example": "1 Example", "five_example": "5 Examples", "ten_example": "10 Examples"}
    if output_dir is None:
        output_dir = os.path.join("data/evaluation", "few_shot_metrics")

    print(f"\n{'=' * 70}\nSTARTING FEW-SHOT EVALUATION ({run_label})\nFolders: {folders}\n{'=' * 70}")

    all_results = {g: {v: None for v in few_shot_variants} for g in st.GROUPS}

    fs_csv_path = os.path.join(
        "data/evaluation", f"few_shot_evaluation{csv_suffix}_{optuna_strategy}.csv"
    )

    def load_fs_csv(path):
        if not os.path.exists(path):
            return False
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                g, v = row['Group'], row['Variant']
                if g in all_results:
                    all_results[g][v] = {
                        "nlp": {
                            "Non-Standardized": {k: float(row[f"nlp_Non-Standardized_{k}"]) for k in ["P", "R", "F1", "IoU"]},
                            "Standardized": {k: float(row[f"nlp_Standardized_{k}"]) for k in ["P", "R", "F1", "IoU"]},
                        },
                        "macro": {
                            m: {f"{fn}_{metric}": float(row[f"macro_{m}_{fn}_{metric}"])
                                for fn in st.MACRO_FIELDS for metric in ["MAE", "MAE_%"]}
                            for m in ["overall", "exact_match"]
                        }
                    }
        return True

    loaded_fs = load_fs_csv(fs_csv_path)
    updated_fs = False

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.endswith('.json') or 'optuna' in filename.lower():
                continue
            pred_path = os.path.join(folder_path, filename)
            g, v_type = parse_model_name(filename)
            if g not in st.GROUPS:
                continue

            prefix = folder_to_prefix[folder]
            if "v2hybrid" in v_type.lower() or "Hybrid v2" in v_type:
                var_name = f"{prefix} (Hybrid v2)"
            elif "Hybrid" in v_type:
                var_name = f"{prefix} (Hybrid v1)"
            elif "Two-Step" in v_type:
                var_name = f"{prefix} (Two-Step)"
            elif "yolo" in v_type.lower():
                var_name = f"{prefix} (YOLO)"
            else:
                var_name = f"{prefix} (Standard)"

            is_yolo = "YOLO" in var_name
            is_standard = var_name.endswith("(Standard)")
            optuna_variant = "YOLO Optuna" if is_yolo else "Standard Optuna"
            var_name_optuna = f"{prefix} ({optuna_variant})"

            needs_base = not (loaded_fs and all_results[g].get(var_name) is not None)
            needs_optuna = (is_yolo or is_standard) and not (
                loaded_fs and all_results[g].get(var_name_optuna) is not None
            )

            if not needs_base and not needs_optuna:
                continue

            try:
                if needs_base:
                    print(f"Evaluating Few-Shot: {var_name} - {g}")
                    evaluator = Nutrition5kEvaluator(pred_path, resources.nlp_model, resources.vocabulary, resources.vocab_embeddings, threshold)
                    all_results[g][var_name] = {
                        "nlp": evaluator.evaluate_nlp_metrics(),
                        "macro": evaluator.evaluate_macronutrients(),
                    }
                    updated_fs = True

                if needs_optuna:
                    print(f"Evaluating Few-Shot: {var_name_optuna} - {g}")
                    optuna_scalars, optuna_summary_path = _load_optuna_scalars_for_model(
                        folder, filename, optuna_strategy
                    )
                    if optuna_scalars:
                        evaluator_opt = Nutrition5kEvaluator(
                            pred_path, resources.nlp_model, resources.vocabulary,
                            resources.vocab_embeddings, threshold,
                            optuna_scalars=optuna_scalars
                        )
                        all_results[g][var_name_optuna] = {
                            "nlp": evaluator_opt.evaluate_nlp_metrics(),
                            "macro": evaluator_opt.evaluate_macronutrients(),
                        }
                        updated_fs = True
                    else:
                        print(f"[!] Could not find unambiguous optuna scalars for {filename} in {optuna_summary_path}")
            except Exception as e:
                print(f"[!] Error in {pred_path}: {e}")

    def save_fs_csv(path):
        print(f"Saving {path}...")
        with open(path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['Group', 'Variant']
            for std_key in ["Non-Standardized", "Standardized"]:
                fieldnames += [f"nlp_{std_key}_{k}" for k in ["P", "R", "F1", "IoU"]]
            for m_key in ["overall", "exact_match"]:
                fieldnames += [f"macro_{m_key}_{fn}_{metric}" for fn in st.MACRO_FIELDS for metric in ["MAE", "MAE_%"]]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for grp in st.GROUPS:
                for var, res in all_results[grp].items():
                    if not res:
                        continue
                    row = {'Group': grp, 'Variant': var}
                    for std_key in ["Non-Standardized", "Standardized"]:
                        for k in ["P", "R", "F1", "IoU"]:
                            row[f"nlp_{std_key}_{k}"] = res["nlp"][std_key][k]
                    for m_key in ["overall", "exact_match"]:
                        for fn in st.MACRO_FIELDS:
                            for metric in ["MAE", "MAE_%"]:
                                row[f"macro_{m_key}_{fn}_{metric}"] = res["macro"][m_key][f"{fn}_{metric}"]
                    writer.writerow(row)

    if updated_fs:
        save_fs_csv(fs_csv_path)

    # --- Charts (few-shot style only) ---
    nlp_dir = os.path.join(output_dir, "nlp_metrics")
    mae_dir = os.path.join(output_dir, "mae_metrics")
    for std_key, suffix in [("Standardized", "std"), ("Non-Standardized", "non_std")]:
        plots.plot_few_shot_nlp(
            {g: all_results[g] for g in st.GROUPS}, nlp_dir, std_key, suffix
        )

    macro_specs = [
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Overall)", "overall", True, False),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Overall)", "overall", True, False),
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Exact Match)", "exact_match", True, False),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Exact Match)", "exact_match", True, False),
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Overall)", "overall", True, True),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Overall)", "overall", True, True),
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Exact Match)", "exact_match", True, True),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Exact Match)", "exact_match", True, True),
    ]
    for metric_suffix, file_suffix, ylabel, title_prefix, subset, plot_baseline, log_scale in macro_specs:
        plots.plot_few_shot_macro(all_results, mae_dir, metric_suffix, file_suffix, ylabel,
                                   title_prefix, subset, plot_baseline, log_scale)

    print(f"Few-shot charts saved in {output_dir}")

    from . import tables
    tables.print_few_shot_summary_tables(all_results, few_shot_variants)
    tables.print_prompting_technique_comparison_table(all_results)

    comparison_dir = os.path.join(output_dir, "prompting_comparison")
    comparison_specs = [
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Overall)", "overall", True, False),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Overall)", "overall", True, False),
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Exact Match)", "exact_match", False, False),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Exact Match)", "exact_match", False, False),
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Overall)", "overall", True, True),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Overall)", "overall", True, True),
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Exact Match)", "exact_match", False, True),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Exact Match)", "exact_match", False, True),
    ]
    for metric_suffix, file_suffix, ylabel, title_prefix, subset, plot_baseline, log_scale in comparison_specs:
        plots.plot_yolo_optuna_metrics(
            all_results, comparison_dir, metric_suffix, file_suffix, ylabel, title_prefix,
            subset, plot_baseline, log_scale,
            strategies=['Standard', 'Standard Optuna', 'YOLO', 'YOLO Optuna'],
            comparison_label="Standard vs Standard Optuna vs YOLO vs YOLO Optuna",
            file_prefix="prompting_comparison",
        )

    return all_results


# ---------------------------------------------------------------------------
# 2) Optuna strategy comparison (dot-plot, few-shot style)
# ---------------------------------------------------------------------------
def evaluate_optuna_strategy_comparison(resources, threshold,
                                         semantic_matching=False, strategies=None, base_dir="data/prediction"):
    strategies = strategies or st.OPTUNA_STRATEGIES
    run_label = "Semantic Matching" if semantic_matching else "non-Semantic Matching"
    print(f"\n{'=' * 70}\nOPTUNA STRATEGY COMPARISON ({run_label})\nStrategies: {strategies}\n{'=' * 70}")

    if semantic_matching:
        folders = ["one_example_semantic_matching", "five_example_semantic_matching", "ten_example_semantic_matching"]
        folder_to_prefix = {"one_example_semantic_matching": "1 Example", "five_example_semantic_matching": "5 Examples",
                             "ten_example_semantic_matching": "10 Examples"}
        cache_path = "data/evaluation/optuna_strategy_comparison_semantic_matching.json"
        output_dir = "data/evaluation/optuna_strategy_metrics_semantic_matching"
    else:
        folders = ["one_example", "five_example", "ten_example"]
        folder_to_prefix = {"one_example": "1 Example", "five_example": "5 Examples", "ten_example": "10 Examples"}
        cache_path = "data/evaluation/optuna_strategy_comparison.json"
        output_dir = "data/evaluation/optuna_strategy_metrics"

    results = {g: {} for g in st.GROUPS}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"Cache loaded from {cache_path}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[!] Corrupted cache ({e}), it will be recreated.")
            results = {g: {} for g in st.GROUPS}

    updated = False

    for folder in folders:
        prefix = folder_to_prefix[folder]
        pred_folder = os.path.join(base_dir, folder)
        if not os.path.exists(pred_folder):
            print(f"[!] Predictions folder not found: {pred_folder}")
            continue

        # Local cache of the Optuna summaries for this folder
        _optuna_cache = {}

        for filename in sorted(os.listdir(pred_folder)):
            if not filename.endswith('.json') or 'optuna' in filename.lower():
                continue
            is_yolo = 'yolo' in filename.lower()
            # For Standard, also exclude the hybrid/twostep files
            if not is_yolo and any(x in filename.lower() for x in ('hybrid', 'twostep')):
                continue

            g, _ = parse_model_name(filename)
            if g not in st.GROUPS:
                continue

            pred_path = os.path.join(pred_folder, filename)
            results.setdefault(g, {})

            if is_yolo:
                # --- YOLO base ---
                yolo_base_var = f"{prefix} ({st.YOLO_BASE_KEY})"
                if results[g].get(yolo_base_var) is None:
                    print(f"  Evaluating [YOLO base] {yolo_base_var} - {g}")
                    try:
                        ev = Nutrition5kEvaluator(pred_path, resources.nlp_model, resources.vocabulary,
                                                  resources.vocab_embeddings, threshold)
                        results[g][yolo_base_var] = {"macro": ev.evaluate_macronutrients()}
                        updated = True
                    except Exception as e:
                        print(f"[!] YOLO base error on {pred_path}: {e}")

                # --- YOLO Optuna for each strategy ---
                for strategy in strategies:
                    var_name = f"{prefix} ({strategy})"
                    if results.get(g, {}).get(var_name) is not None:
                        continue
                    optuna_scalars, summary_path = _load_optuna_scalars_for_model(folder, filename, strategy)
                    if not optuna_scalars:
                        print(f"[!] YOLO scalars not found for {filename} / {strategy} in {summary_path}")
                        continue
                    print(f"  Evaluating [YOLO {strategy}] {var_name} - {g}")
                    try:
                        ev = Nutrition5kEvaluator(pred_path, resources.nlp_model, resources.vocabulary,
                                                  resources.vocab_embeddings, threshold,
                                                  optuna_scalars=optuna_scalars)
                        results[g][var_name] = {"macro": ev.evaluate_macronutrients()}
                        updated = True
                    except Exception as e:
                        print(f"[!] YOLO {strategy} error on {pred_path}: {e}")

            else:
                # --- Standard base ---
                std_base_var = f"{prefix} ({st.STANDARD_BASE_KEY})"
                if results[g].get(std_base_var) is None:
                    print(f"  Evaluating [Standard base] {std_base_var} - {g}")
                    try:
                        ev = Nutrition5kEvaluator(pred_path, resources.nlp_model, resources.vocabulary,
                                                  resources.vocab_embeddings, threshold)
                        results[g][std_base_var] = {"macro": ev.evaluate_macronutrients()}
                        updated = True
                    except Exception as e:
                        print(f"[!] Standard base error on {pred_path}: {e}")

                # --- Standard Optuna for each strategy ---
                for strategy in strategies:
                    var_name = f"{prefix} ({strategy}_std)"
                    if results.get(g, {}).get(var_name) is not None:
                        continue
                    optuna_scalars, summary_path = _load_optuna_scalars_for_model(folder, filename, strategy)
                    if not optuna_scalars:
                        print(f"[!] Standard scalars not found for {filename} / {strategy} in {summary_path}")
                        continue
                    print(f"  Evaluating [Standard {strategy}] {var_name} - {g}")
                    try:
                        ev = Nutrition5kEvaluator(pred_path, resources.nlp_model, resources.vocabulary,
                                                  resources.vocab_embeddings, threshold,
                                                  optuna_scalars=optuna_scalars)
                        results[g][var_name] = {"macro": ev.evaluate_macronutrients()}
                        updated = True
                    except Exception as e:
                        print(f"[!] Standard {strategy} error on {pred_path}: {e}")

    if updated:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"Cache saved in {cache_path}")

    os.makedirs(output_dir, exist_ok=True)
    specs = [
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Optuna Strategies - Absolute MAE (Overall)", "overall", True, False),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Optuna Strategies - Percentage MAE (Overall)", "overall", True, False),
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Optuna Strategies - Absolute MAE (Exact Match)", "exact_match", False, False),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Optuna Strategies - Percentage MAE (Exact Match)", "exact_match", False, False),
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Optuna Strategies - Absolute MAE (Overall)", "overall", True, True),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Optuna Strategies - Percentage MAE (Overall)", "overall", True, True),
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Optuna Strategies - Absolute MAE (Exact Match)", "exact_match", False, True),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Optuna Strategies - Percentage MAE (Exact Match)", "exact_match", False, True),
    ]
    for metric_suffix, file_suffix, ylabel, title_prefix, subset, plot_baseline, log_scale in specs:
        plots.plot_optuna_strategy_metrics(results, output_dir, metric_suffix, file_suffix, ylabel,
                                            title_prefix, subset, plot_baseline, log_scale)

    print(f"Optuna Strategy charts saved in {output_dir}")

    from . import tables
    tables.print_optuna_strategy_tables(results, strategies)

    return results


# ---------------------------------------------------------------------------
# 3) YOLO vs YOLO Optuna comparison (dot-plot, few-shot style)
# ---------------------------------------------------------------------------
def evaluate_yolo_optuna_comparison(resources, threshold,
                                     base_dir="data/prediction", semantic_matching=False, optuna_strategy="linear_all5"):
    run_label = "Semantic Matching" if semantic_matching else "Normal"
    print(f"\n{'=' * 70}\nSTARTING YOLO OPTUNA EVALUATION COMPARISON ({run_label}, strategy={optuna_strategy})\n{'=' * 70}")

    if semantic_matching:
        folders = ["one_example_semantic_matching", "five_example_semantic_matching", "ten_example_semantic_matching"]
        folder_to_prefix = {"one_example_semantic_matching": "1 Example", "five_example_semantic_matching": "5 Examples",
                             "ten_example_semantic_matching": "10 Examples"}
        yolo_optuna_json_path = f"data/evaluation/few_shot_yolo_optuna_evaluation_semantic_matching_{optuna_strategy}.json"
        output_dir = f"data/evaluation/yolo_optuna_metrics_semantic_matching_{optuna_strategy}"
    else:
        folders = ["one_example", "five_example", "ten_example"]
        folder_to_prefix = {"one_example": "1 Example", "five_example": "5 Examples", "ten_example": "10 Examples"}
        yolo_optuna_json_path = f"data/evaluation/few_shot_yolo_optuna_evaluation_{optuna_strategy}.json"
        output_dir = f"data/evaluation/yolo_optuna_metrics_{optuna_strategy}"

    results = {}
    if os.path.exists(yolo_optuna_json_path):
        print(f"Loading existing YOLO Optuna evaluation from {yolo_optuna_json_path}")
        try:
            with open(yolo_optuna_json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[!] Corrupted or empty cache file ({e}), it will be recreated from scratch.")
            os.remove(yolo_optuna_json_path)
            results = {}

    updated = False

    for folder in folders:
        pred_folder = os.path.join(base_dir, folder)
        optuna_summary_path = os.path.join("data", "optuna", "few_shot_yolo", folder,
                                             optuna_strategy, "optuna_summary.json")
        if not os.path.exists(pred_folder) or not os.path.exists(optuna_summary_path):
            print(f"[!] Skipping {folder}: pred_folder or optuna_summary not found (strategy='{optuna_strategy}')")
            continue

        with open(optuna_summary_path, 'r', encoding='utf-8') as f:
            optuna_data = json.load(f)

        for filename in os.listdir(pred_folder):
            if not filename.endswith('.json') or 'yolo' not in filename.lower():
                continue
            g, _ = parse_model_name(filename)
            if g not in st.GROUPS:
                continue

            prefix = folder_to_prefix[folder]
            var_name = f"{prefix} (YOLO)"
            var_name_optuna = f"{prefix} (YOLO Optuna)"
            results.setdefault(g, {})

            if var_name_optuna in results[g] and var_name in results[g]:
                continue
            updated = True

            model_identifier = filename.replace('test_predictions_nutrition5k_yolo_', '').replace('.json', '')
            optuna_scalars = None
            for key, val in optuna_data.items():
                if key.startswith("_"):
                    continue
                if model_identifier in key:
                    optuna_scalars = val.get("optimal_scalars")
                    break
            if not optuna_scalars:
                print(f"[!] Could not find optuna scalars for {filename} in {optuna_summary_path}")
                continue

            print(f"Evaluating YOLO Optuna: {var_name_optuna} - {g}")
            pred_path = os.path.join(pred_folder, filename)

            if var_name not in results[g]:
                evaluator_base = Nutrition5kEvaluator(pred_path, resources.nlp_model, resources.vocabulary, resources.vocab_embeddings, threshold)
                results[g][var_name] = {
                    "nlp": evaluator_base.evaluate_nlp_metrics(),
                    "macro": evaluator_base.evaluate_macronutrients(),
                }

            evaluator_opt = Nutrition5kEvaluator(pred_path, resources.nlp_model, resources.vocabulary, resources.vocab_embeddings, threshold,
                                                  optuna_scalars=optuna_scalars)
            results[g][var_name_optuna] = {
                "nlp": evaluator_opt.evaluate_nlp_metrics(),
                "macro": evaluator_opt.evaluate_macronutrients(),
            }

    if updated:
        with open(yolo_optuna_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

    os.makedirs(output_dir, exist_ok=True)
    specs = [
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Overall)", "overall", True, False),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Overall)", "overall", True, False),
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Exact Match)", "exact_match", False, False),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Exact Match)", "exact_match", False, False),
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Overall)", "overall", True, True),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Overall)", "overall", True, True),
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Exact Match)", "exact_match", False, True),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Exact Match)", "exact_match", False, True),
    ]
    for metric_suffix, file_suffix, ylabel, title_prefix, subset, plot_baseline, log_scale in specs:
        plots.plot_yolo_optuna_metrics(results, output_dir, metric_suffix, file_suffix, ylabel,
                                        title_prefix, subset, plot_baseline, log_scale)

    print(f"YOLO Optuna charts saved in {output_dir}")
    return results


# ---------------------------------------------------------------------------
# 4) Threshold study (threshold loop over all the one-example models)
# ---------------------------------------------------------------------------
def evaluate_threshold_study(resources, pred_json_paths,
                              thresholds, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    study_json_path = os.path.join(output_dir, "threshold_study_results.json")

    if os.path.exists(study_json_path):
        print(f"Loading existing threshold study from {study_json_path}...")
        with open(study_json_path, 'r', encoding='utf-8') as f:
            threshold_study_results = json.load(f)
    else:
        threshold_study_results = {}

    updated = False
    for pred_path in pred_json_paths:
        model_name = os.path.basename(pred_path).replace('test_predictions_nutrition5k_', '').replace('.json', '')
        if model_name in threshold_study_results:
            continue

        print(f"Computing threshold study for missing model: {model_name}...")
        updated = True
        threshold_study_results[model_name] = {'F1': [], 'IoU': []}

        try:
            evaluator = Nutrition5kEvaluator(pred_path, resources.nlp_model, resources.vocabulary, resources.vocab_embeddings, threshold=1.0)
            for thr in thresholds:
                evaluator.threshold = thr
                nlp_res = evaluator.evaluate_nlp_metrics()
                if nlp_res and "Standardized" in nlp_res:
                    threshold_study_results[model_name]['F1'].append(nlp_res["Standardized"]["F1"])
                    threshold_study_results[model_name]['IoU'].append(nlp_res["Standardized"]["IoU"])
                else:
                    threshold_study_results[model_name]['F1'].append(0.0)
                    threshold_study_results[model_name]['IoU'].append(0.0)
        except Exception as e:
            print(f"[!] Error during the threshold study of file {pred_path}: {e}")
            threshold_study_results.pop(model_name, None)

    if updated:
        with open(study_json_path, 'w', encoding='utf-8') as f:
            json.dump(threshold_study_results, f, indent=4)

    plots.plot_threshold_study(thresholds, threshold_study_results, output_dir, parse_model_name)
    return threshold_study_results


# ---------------------------------------------------------------------------
# 5) Per-model evaluation (one_example): feeds the NLP metrics plot
#    (bar chart) and the summary tables. Uses the same CSV cache
#    (nlp_evaluation.csv): if a model is already present it is not
#    recomputed, so the NLP model may never be
#    loaded.
# ---------------------------------------------------------------------------
def evaluate_single_run_models(resources, threshold, pred_json_paths, eval_csv_path, similarity_dir=None, optuna_strategy="linear_all5"):
    all_nlp_results = {}
    all_macro_results = {}

    if os.path.exists(eval_csv_path):
        print(f"Loading existing evaluation results from {eval_csv_path}...")
        with open(eval_csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                m_name = row['model_name']
                all_nlp_results[m_name] = {
                    "Non-Standardized": {k: float(row[f"nlp_Non-Standardized_{k}"]) for k in ["P", "R", "F1", "IoU"]},
                    "Standardized": {k: float(row[f"nlp_Standardized_{k}"]) for k in ["P", "R", "F1", "IoU"]},
                }
                all_macro_results[m_name] = {"overall": {}, "exact_match": {}}
                for fn in st.MACRO_FIELDS:
                    for m_key in ["overall", "exact_match"]:
                        all_macro_results[m_name][m_key][f"{fn}_MAE"] = float(row[f"macro_{m_key}_{fn}_MAE"])
                        all_macro_results[m_name][m_key][f"{fn}_MAE_%"] = float(row[f"macro_{m_key}_{fn}_MAE_%"])

    updated = False

    for pred_path in pred_json_paths:
        model_name = os.path.basename(pred_path).replace('test_predictions_nutrition5k_', '').replace('.json', '')
        is_yolo = "yolo" in model_name.lower()
        is_standard = "optuna" not in model_name.lower() and not any(
            x in model_name.lower() for x in ("twostep", "hybrid")
        ) and not is_yolo
        optuna_label = "YOLO Optuna" if is_yolo else "Standard Optuna"
        model_name_optuna = model_name + "_optuna"

        needs_base = model_name not in all_nlp_results or model_name not in all_macro_results
        needs_optuna = (is_yolo or is_standard) and (
            model_name_optuna not in all_nlp_results or model_name_optuna not in all_macro_results
        )

        if not needs_base and not needs_optuna:
            continue

        print(f"\n{'=' * 70}\nMODEL EVALUATION: {model_name}\n{'=' * 70}")
        updated = True

        try:
            if needs_base:
                evaluator = Nutrition5kEvaluator(pred_path, resources.nlp_model, resources.vocabulary,
                                                  resources.vocab_embeddings, threshold)
                nlp_res = evaluator.evaluate_nlp_metrics()
                all_nlp_results[model_name] = nlp_res

                if similarity_dir:
                    os.makedirs(similarity_dir, exist_ok=True)
                    sim_path = os.path.join(similarity_dir, f"similarity_analysis_{model_name}.json")
                    evaluator.export_similarity_log_to_json(sim_path)
                    evaluator.print_similarity_summary()

                all_macro_results[model_name] = evaluator.evaluate_macronutrients()

            if needs_optuna:
                print(f"--- Evaluating {optuna_label} variant: {model_name_optuna} ---")
                optuna_scalars, optuna_summary_path = _load_optuna_scalars_for_model(
                    "one_example", os.path.basename(pred_path), optuna_strategy
                )
                if optuna_scalars:
                    evaluator_opt = Nutrition5kEvaluator(
                        pred_path, resources.nlp_model, resources.vocabulary,
                        resources.vocab_embeddings, threshold,
                        optuna_scalars=optuna_scalars
                    )
                    all_nlp_results[model_name_optuna] = evaluator_opt.evaluate_nlp_metrics()
                    all_macro_results[model_name_optuna] = evaluator_opt.evaluate_macronutrients()
                else:
                    print(f"[!] Could not find unambiguous optuna scalars for {model_name} in {optuna_summary_path}")
        except Exception as e:
            print(f"\n[!] Error during the execution of file {pred_path}:\n{e}")

    if updated:
        print(f"Saving evaluation results to {eval_csv_path}...")
        os.makedirs(os.path.dirname(eval_csv_path) or ".", exist_ok=True)
        with open(eval_csv_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['model_name']
            for std_key in ["Non-Standardized", "Standardized"]:
                fieldnames += [f"nlp_{std_key}_{k}" for k in ["P", "R", "F1", "IoU"]]
            for m_key in ["overall", "exact_match"]:
                for fn in st.MACRO_FIELDS:
                    fieldnames += [f"macro_{m_key}_{fn}_MAE", f"macro_{m_key}_{fn}_MAE_%"]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m_name in all_nlp_results:
                if m_name not in all_macro_results:
                    continue
                row = {'model_name': m_name}
                for std_key in ["Non-Standardized", "Standardized"]:
                    for k in ["P", "R", "F1", "IoU"]:
                        row[f"nlp_{std_key}_{k}"] = all_nlp_results[m_name][std_key][k]
                for m_key in ["overall", "exact_match"]:
                    for fn in st.MACRO_FIELDS:
                        row[f"macro_{m_key}_{fn}_MAE"] = all_macro_results[m_name][m_key][f"{fn}_MAE"]
                        row[f"macro_{m_key}_{fn}_MAE_%"] = all_macro_results[m_name][m_key][f"{fn}_MAE_%"]
                writer.writerow(row)

    return all_nlp_results, all_macro_results


def evaluate_prompting_technique_comparison(few_shot_results: dict, output_dir: str,
                                             comparison_label: str = "Standard vs YOLO vs YOLO Optuna"):
    os.makedirs(output_dir, exist_ok=True)
    strategies = ['Standard', 'Standard Optuna', 'YOLO', 'YOLO Optuna']

    specs = [
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Overall)", "overall", True, False),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Overall)", "overall", True, False),
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Exact Match)", "exact_match", False, False),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Exact Match)", "exact_match", False, False),
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Overall)", "overall", True, True),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Overall)", "overall", True, True),
        ("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Exact Match)", "exact_match", False, True),
        ("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Exact Match)", "exact_match", False, True),
    ]
    for metric_suffix, file_suffix, ylabel, title_prefix, subset, plot_baseline, log_scale in specs:
        plots.plot_yolo_optuna_metrics(
            few_shot_results, output_dir, metric_suffix, file_suffix, ylabel, title_prefix,
            subset, plot_baseline, log_scale,
            strategies=strategies, comparison_label=comparison_label,
            file_prefix="prompting_comparison",
        )

    print(f"Prompting technique comparison charts saved in {output_dir}")
    return None


def evaluate_all_strategies_combined(resources, threshold, output_dir, optuna_strategy="linear_all5"):
    """
    Reads the CSVs produced by evaluate_few_shot_learning (normal and semantic_matching) and generates
    8 combined charts (same structure as the prompting_comparison charts)
    combining the 8 strategies:
      Standard / Standard Optuna / YOLO / YOLO Optuna           (normal files)
      Standard Matching / Standard Optuna Matching /             (Semantic Matching files)
      YOLO Matching / YOLO Optuna Matching

    Legend: only the 4 base variants (without Optuna) plus the meaning of the lines
    (solid = classic, dashed = Optuna). The lighter color indicates
    Semantic Matching.
    """
    import os
    import csv as _csv
    import matplotlib
    matplotlib.use('Agg')
    from . import style as st
    from .plots import plot_yolo_optuna_metrics

    base_eval_dir = "data/evaluation"
    csv_normal = os.path.join(base_eval_dir, f"few_shot_evaluation_{optuna_strategy}.csv")
    csv_semantic_matching = os.path.join(base_eval_dir, f"few_shot_evaluation_semantic_matching_{optuna_strategy}.csv")

    # Map from the CSV name to the strategy name in the result
    NORMAL_MAP = {
        'Standard':        'Standard',
        'Standard Optuna': 'Standard Optuna',
        'YOLO':            'YOLO',
        'YOLO Optuna':     'YOLO Optuna',
    }
    SEMANTIC_MATCHING_MAP = {
        'Standard':        'Standard Matching',
        'Standard Optuna': 'Standard Optuna Matching',
        'YOLO':            'YOLO Matching',
        'YOLO Optuna':     'YOLO Optuna Matching',
    }

    all_results = {g: {} for g in st.GROUPS}

    def process_csv(csv_path, strategy_map):
        if not os.path.exists(csv_path):
            print(f"[WARNING] CSV file not found: {csv_path}")
            return
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in _csv.DictReader(f):
                g       = row.get('Group', '')
                variant = row.get('Variant', '')
                if g not in st.GROUPS or ' (' not in variant:
                    continue
                prefix, rest = variant.split(' (', 1)
                strat_raw = rest.rstrip(')')
                if strat_raw not in strategy_map:
                    continue
                mapped = strategy_map[strat_raw]
                key    = f"{prefix} ({mapped})"

                macro = {}
                for subset in ('overall', 'exact_match'):
                    macro[subset] = {}
                    for field in st.MACRO_FIELDS:
                        for metric in ('MAE', 'MAE_%'):
                            col = f"macro_{subset}_{field}_{metric}"
                            if col in row and row[col]:
                                macro[subset][f"{field}_{metric}"] = float(row[col])

                all_results[g][key] = {"macro": macro}

    process_csv(csv_normal, NORMAL_MAP)
    process_csv(csv_semantic_matching, SEMANTIC_MATCHING_MAP)

    if not any(v for v in all_results.values()):
        print("[WARNING] No valid data found to generate the combined charts.")
        return

    os.makedirs(output_dir, exist_ok=True)

    strategies = [
        'Standard', 'Standard Optuna',
        'Standard Matching', 'Standard Optuna Matching',
        'YOLO', 'YOLO Optuna',
        'YOLO Matching', 'YOLO Optuna Matching',
    ]
    # The legend shows only the 4 base variants (without Optuna): the meaning of
    # solid/dashed is explained by the line entries, the light color by Matching.
    legend_strats = ['Standard', 'Standard Matching', 'YOLO', 'YOLO Matching']

    specs = [
        ("MAE",   "mae_abs",  "MAE (Value)", "Absolute MAE (Overall)",       "overall",     True,  False),
        ("MAE_%", "mae_perc", "MAE (%)",     "Percentage MAE (Overall)",      "overall",     True,  False),
        ("MAE",   "mae_abs",  "MAE (Value)", "Absolute MAE (Exact Match)",    "exact_match", False, False),
        ("MAE_%", "mae_perc", "MAE (%)",     "Percentage MAE (Exact Match)",  "exact_match", False, False),
        ("MAE",   "mae_abs",  "MAE (Value)", "Absolute MAE (Overall)",        "overall",     True,  True),
        ("MAE_%", "mae_perc", "MAE (%)",     "Percentage MAE (Overall)",      "overall",     True,  True),
        ("MAE",   "mae_abs",  "MAE (Value)", "Absolute MAE (Exact Match)",    "exact_match", False, True),
        ("MAE_%", "mae_perc", "MAE (%)",     "Percentage MAE (Exact Match)",  "exact_match", False, True),
    ]
    for ms, fs, yl, tp, sub, pb, ls in specs:
        plot_yolo_optuna_metrics(
            all_results, output_dir, ms, fs, yl, tp,
            subset=sub, plot_baseline=pb, log_scale=ls,
            strategies=strategies,
            comparison_label="Strategy Comparison: Standard / YOLO / Semantic Matching",
            file_prefix="combined_strategies",
            legend_strategies=legend_strats,
        )

    print(f"[INFO] 8 combined charts generated in: {output_dir}")