import json
import optuna
import os
import glob

# ================= CONFIGURAZIONI =================
# 1. PERCORSI
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "data", "ground truth", "train_set_optuna.json")
PREDICTIONS_BASE_DIR = os.path.join(BASE_DIR, "data", "optuna")


# ==================================================

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# Carichiamo la ground truth una volta sola all'avvio
print("Caricamento Ground Truth...")
ground_truth_data = load_json(GROUND_TRUTH_FILE)
gt_dict = {item['food_id']: item for item in ground_truth_data}


def optimize_for_file(predictions_file):
    predictions_data = load_json(predictions_file)

    best_params = {}
    total_mae_perc = 0.0

    # Analizziamo un nutriente alla volta in modo completamente indipendente
    nutrients = ["mass", "cal", "fat", "carbs", "protein"]

    for nut in nutrients:
        def objective(trial):
            # Range di ricerca impostato tra 0.05 e 4.0 per la strategia solo scalari
            scale = trial.suggest_float(f"scale_{nut}", 0.005, 5.0)

            errors = []
            gt_total = 0.0

            for pred in predictions_data:
                food_id = pred.get("food_id")
                if food_id not in gt_dict:
                    continue

                gt = gt_dict[food_id]
                gt_nut = gt.get("total_macronutrients", {})

                # Recupero Ground Truth specifico per il nutriente corrente
                if nut == "mass":
                    gt_val = gt_nut.get("total_mass_grams", gt.get("total_mass", 0))
                elif nut == "cal":
                    gt_val = gt_nut.get("calories", gt.get("total_calories", 0))
                elif nut == "fat":
                    gt_val = gt_nut.get("fat", gt.get("total_fat", 0))
                elif nut == "carbs":
                    gt_val = gt_nut.get("carbohydrates", gt.get("total_carbs", 0))
                elif nut == "protein":
                    gt_val = gt_nut.get("protein", gt.get("total_protein", 0))

                # Evita problemi se la massa è assente nel ground truth
                if gt_val <= 0 and nut == "mass":
                    continue

                gt_total += gt_val

                # Recupero Predizione specifica
                pred_block = pred.get("predicted_nutrition", pred.get("predicted_nutrition_step2", {}))
                pred_nut_dict = pred_block.get("total_macronutrients", {})

                if nut == "mass":
                    ingredients = pred_block.get("visible_ingredients", []) + pred_block.get("hidden_ingredients", [])
                    pred_val = sum(ing.get("weight_grams", 0) for ing in ingredients)
                elif nut == "cal":
                    pred_val = pred_nut_dict.get("calories", 0)
                elif nut == "fat":
                    pred_val = pred_nut_dict.get("fat", 0)
                elif nut == "carbs":
                    pred_val = pred_nut_dict.get("carbohydrates", 0)
                elif nut == "protein":
                    pred_val = pred_nut_dict.get("protein", 0)

                # Applicazione correzione: solo moltiplicatore (niente bias)
                adj_val = max(0.0, pred_val * scale)
                errors.append(abs(adj_val - gt_val))

            # Controllo di sicurezza contro divisioni per zero
            if gt_total == 0 or len(errors) == 0:
                return 9999.0

            # Ritorna il MAE % specifico del singolo nutriente
            return (sum(errors) / gt_total) * 100

        # Silenziamo l'output di Optuna per non inondare la console
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")

        # Eseguiamo 2000 tentativi mirati a trovare il valore ottimale per il singolo nutriente
        study.optimize(objective, n_trials=5000)

        # Salviamo il parametro scalare migliore trovato e sommiamo l'errore per la media finale
        best_params.update(study.best_params)
        total_mae_perc += study.best_value

    # Restituiamo la media aritmetica dei 5 MAE indipendenti e il dizionario combinato dei pesi ottimali
    return total_mae_perc / 5, best_params


if __name__ == "__main__":
    search_pattern = os.path.join(PREDICTIONS_BASE_DIR, "**", "*.json")
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        print(f"Nessun file JSON trovato in {PREDICTIONS_BASE_DIR}. Controlla il percorso.")

    folder_results = {}

    print(f"\nAvvio ottimizzazione Optuna Indipendente (Solo Moltiplicatori: 0.05x - 4.0x)\n")

    for filepath in sorted(json_files):
        # Ignoriamo i file di summary già presenti per evitare di ri-analizzarli
        if os.path.basename(filepath) == "optuna_summary.json":
            continue

        print(f"-> Ottimizzazione in corso per: {filepath}")
        best_mae, best_params = optimize_for_file(filepath)
        print(f"   MAE% Medio Combinato Raggiunto: {best_mae:.2f}%")

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