import json
import optuna
import os
import glob

# ================= CONFIGURAZIONI =================
# Variabile per decidere quale strategia di ottimizzazione usare:
# False -> Solo moltiplicatori (Strategia originale)
# True  -> Moltiplicatori + Intercetta (Scala + Bias)
USE_BIAS_STRATEGY = True
# ==================================================

# 1. PERCORSI
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "data", "ground truth", "train_set_optuna.json")
PREDICTIONS_BASE_DIR = os.path.join(BASE_DIR, "data", "optuna")


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# Carichiamo la ground truth una volta sola
print("Caricamento Ground Truth...")
ground_truth_data = load_json(GROUND_TRUTH_FILE)
gt_dict = {item['food_id']: item for item in ground_truth_data}


def optimize_for_file(predictions_file, use_bias):
    predictions_data = load_json(predictions_file)

    def objective(trial):
        # 1. Moltiplicatori (sempre usati in entrambe le strategie)
        scale_mass = trial.suggest_float("scale_mass", 0.3, 2.0)
        scale_cal = trial.suggest_float("scale_cal", 0.3, 2.0)
        scale_fat = trial.suggest_float("scale_fat", 0.3, 2.0)
        scale_carbs = trial.suggest_float("scale_carbs", 0.3, 2.0)
        scale_protein = trial.suggest_float("scale_protein", 0.3, 2.0)

        # 2. Intercette (usate solo se la strategia Bias è attiva)
        if use_bias:
            bias_mass = trial.suggest_float("bias_mass", -100.0, 100.0)
            bias_cal = trial.suggest_float("bias_cal", -100.0, 100.0)
            bias_fat = trial.suggest_float("bias_fat", -50.0, 50.0)
            bias_carbs = trial.suggest_float("bias_carbs", -50.0, 50.0)
            bias_protein = trial.suggest_float("bias_protein", -50.0, 50.0)
        else:
            bias_mass = 0.0
            bias_cal = 0.0
            bias_fat = 0.0
            bias_carbs = 0.0
            bias_protein = 0.0

        errors = {"mass": [], "cal": [], "fat": [], "carbs": [], "protein": []}
        gt_totals = {"mass": 0, "cal": 0, "fat": 0, "carbs": 0, "protein": 0}

        for pred in predictions_data:
            food_id = pred.get("food_id")
            if food_id not in gt_dict:
                continue

            gt = gt_dict[food_id]

            # Lettura corretta dalla struttura annidata total_macronutrients
            gt_nut = gt.get("total_macronutrients", {})
            gt_mass = gt_nut.get("total_mass_grams", gt.get("total_mass", 0))
            gt_cal = gt_nut.get("calories", gt.get("total_calories", 0))
            gt_fat = gt_nut.get("fat", gt.get("total_fat", 0))
            gt_carbs = gt_nut.get("carbohydrates", gt.get("total_carbs", 0))
            gt_protein = gt_nut.get("protein", gt.get("total_protein", 0))

            if gt_mass <= 0:
                continue

            gt_totals["mass"] += gt_mass
            gt_totals["cal"] += gt_cal
            gt_totals["fat"] += gt_fat
            gt_totals["carbs"] += gt_carbs
            gt_totals["protein"] += gt_protein

            # Recupero dati predetti (compatibile con Step 1, Two-Step e Hybrid)
            pred_block = pred.get("predicted_nutrition", pred.get("predicted_nutrition_step2", {}))
            pred_nut = pred_block.get("total_macronutrients", {})

            ingredients = pred_block.get("visible_ingredients", []) + pred_block.get("hidden_ingredients", [])
            pred_mass = sum(ing.get("weight_grams", 0) for ing in ingredients)

            pred_cal = pred_nut.get("calories", 0)
            pred_fat = pred_nut.get("fat", 0)
            pred_carbs = pred_nut.get("carbohydrates", 0)
            pred_protein = pred_nut.get("protein", 0)

            # 3. Applicazione (Scale + Bias con protezione dai negativi)
            adj_mass = max(0.0, (pred_mass * scale_mass) + bias_mass)
            adj_cal = max(0.0, (pred_cal * scale_cal) + bias_cal)
            adj_fat = max(0.0, (pred_fat * scale_fat) + bias_fat)
            adj_carbs = max(0.0, (pred_carbs * scale_carbs) + bias_carbs)
            adj_protein = max(0.0, (pred_protein * scale_protein) + bias_protein)

            errors["mass"].append(abs(adj_mass - gt_mass))
            errors["cal"].append(abs(adj_cal - gt_cal))
            errors["fat"].append(abs(adj_fat - gt_fat))
            errors["carbs"].append(abs(adj_carbs - gt_carbs))
            errors["protein"].append(abs(adj_protein - gt_protein))

        # Controllo di sicurezza contro divisioni per zero
        if gt_totals["mass"] == 0 or len(errors["mass"]) == 0:
            return 9999.0

        mae_perc_mass = (sum(errors["mass"]) / gt_totals["mass"]) * 100
        mae_perc_cal = (sum(errors["cal"]) / gt_totals["cal"]) * 100
        mae_perc_fat = (sum(errors["fat"]) / gt_totals["fat"]) * 100
        mae_perc_carbs = (sum(errors["carbs"]) / gt_totals["carbs"]) * 100
        mae_perc_protein = (sum(errors["protein"]) / gt_totals["protein"]) * 100

        return (mae_perc_mass + mae_perc_cal + mae_perc_fat + mae_perc_carbs + mae_perc_protein) / 5

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2000)

    return study.best_value, study.best_params


if __name__ == "__main__":
    import csv

    search_pattern = os.path.join(PREDICTIONS_BASE_DIR, "**", "*.json")
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        print(f"Nessun file JSON trovato in {PREDICTIONS_BASE_DIR}. Controlla il percorso.")

    folder_results = {}

    strategy_name = "Scala + Intercetta (Bias)" if USE_BIAS_STRATEGY else "Solo Moltiplicatori"
    print(f"\nAvvio ottimizzazione Optuna. Strategia in uso: {strategy_name}\n")

    for filepath in sorted(json_files):
        # Ignoriamo i file di summary già presenti per evitare di ri-analizzarli
        if os.path.basename(filepath) == "optuna_summary.json":
            continue

        print(f"-> Ottimizzazione in corso per: {filepath}")
        best_mae, best_params = optimize_for_file(filepath, USE_BIAS_STRATEGY)
        print(f"   MAE% Raggiunto: {best_mae:.2f}%")

        folder_path = os.path.dirname(filepath)
        config_name = os.path.basename(filepath)

        if folder_path not in folder_results:
            folder_results[folder_path] = {}

        folder_results[folder_path][config_name] = {
            "best_combined_mae_perc": round(best_mae, 2),
            "optimal_scalars": best_params
        }

    # Salvataggio dei file JSON riassuntivi per ogni cartella
    for folder_path, results in folder_results.items():
        json_filename = os.path.join(folder_path, "optuna_summary.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"✅ Salvato riepilogo in {json_filename}")

    print("\n✅ Ottimizzazione batch completata! Tutti i JSON sono stati salvati nelle rispettive cartelle.")