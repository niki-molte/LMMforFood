import json
import optuna
import os
import glob


# ================= CONFIGURAZIONI =================
# 1. PERCORSI
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "data", "ground truth", "train_set_optuna.json")
PREDICTIONS_BASE_DIR = os.path.join(BASE_DIR, "data", "optuna")

# 2. STRATEGIA OPTUNA
# Imposta qui la strategia da usare, oppure tramite variabile d'ambiente OPTUNA_STRATEGY.
# Strategie disponibili (vedi STRATEGY_CATALOG):
#   "linear_all5"      -> 5 scale indipendenti  [default, già eseguita]
#   "linear_mass_only" -> ottimizza solo la massa (1 parametro)
#   "power_law_all5"   -> legge di potenza (scale * pred ^ power) (10 parametri)
#   "piecewise_all5"   -> correzione a soglia per valori piccoli/grandi (15 parametri)
#   "linear_bias_all5" -> modello lineare con scale e bias (10 parametri)
OPTUNA_STRATEGY = os.environ.get("OPTUNA_STRATEGY", "all")

# Catalogo delle strategie
STRATEGY_CATALOG = {
    "linear_all5": {
        "nutrients": ["mass", "cal", "fat", "carbs", "protein"],
        "description": "Moltiplicatore lineare indipendente per ciascuno dei 5 macronutrienti",
        "param_type": "linear_scale",
        "n_trials": 2000,
    },
    "linear_mass_only": {
        "nutrients": ["mass"],
        "description": "Moltiplicatore lineare ottimizzato solo per la massa totale",
        "param_type": "linear_scale",
        "n_trials": 2000,
    },
    "power_law_all5": {
        "nutrients": ["mass", "cal", "fat", "carbs", "protein"],
        "description": "Correzione esponenziale: scale * (pred_val ^ power) per ogni nutriente (10 parametri)",
        "param_type": "power_scale",
        "n_trials": 2000,
    },
    "piecewise_all5": {
        "nutrients": ["mass", "cal", "fat", "carbs", "protein"],
        "description": "Moltiplicatore a soglia: scale_small se pred < threshold, altrimenti scale_large (15 parametri)",
        "param_type": "piecewise_scale",
        "n_trials": 2000,
    },
    "linear_bias_all5": {
        "nutrients": ["mass", "cal", "fat", "carbs", "protein"],
        "description": "Modello lineare con scale e bias (adj_val = pred * scale + bias)",
        "param_type": "linear_scale_bias",
        "n_trials": 2000,
    },
}


# ==================================================

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# Carichiamo la ground truth una volta sola all'avvio
print("Caricamento Ground Truth...")
ground_truth_data = load_json(GROUND_TRUTH_FILE)
gt_dict = {item['food_id']: item for item in ground_truth_data}


def get_gt_val(gt, nut):
    """Estrae il valore ground truth per un dato nutriente."""
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
    """Estrae il valore predetto per un dato nutriente."""
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
    """Strategia: scale lineare indipendente per ogni nutriente."""
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
    """Strategia: Legge di Potenza (esponenziale). adj_val = scale * (pred_val ** power)"""
    nutrients = strategy_cfg["nutrients"]
    n_trials = strategy_cfg["n_trials"]
    best_params = {}
    total_mae_perc = 0.0

    for nut in nutrients:
        def objective(trial, _nut=nut):
            scale = trial.suggest_float(f"scale_{_nut}", 0.05, 5.0)
            power = trial.suggest_float(f"power_{_nut}", 0.3, 2.5)  # Ampio range per la potenza
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

                # Applica la formula non lineare (evita crash se pred_val è esattamente 0)
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
    Calcola il range [min, max] dei valori predetti per un dato nutriente
    nel train set, con un margine percentuale di sicurezza.
    Usato per impostare dinamicamente i bound del parametro threshold
    in optimize_piecewise_all5, invece di un range fisso [1, 1500] uguale per tutti.
    """
    vals = [get_pred_val(pred, nut) for pred in predictions_data]
    vals = [v for v in vals if v > 0]  # scarta valori nulli/non validi

    if not vals:
        # fallback di sicurezza se non ci sono valori validi
        return 1.0, 1500.0

    v_min, v_max = min(vals), max(vals)

    # aggiunge un margine per non tagliare fuori la soglia ottimale
    # se sta vicino agli estremi osservati
    range_span = v_max - v_min
    low = max(0.01, v_min - margin * range_span)
    high = v_max + margin * range_span

    return low, high


def optimize_piecewise_all5(predictions_data, strategy_cfg):
    """Strategia: A soglia (Piecewise). Scale diverso se pred_val è minore o maggiore di un limite."""
    nutrients = strategy_cfg["nutrients"]
    n_trials = strategy_cfg["n_trials"]
    best_params = {}
    total_mae_perc = 0.0

    for nut in nutrients:
        # NUOVO: calcola il range di soglia specifico per questo nutriente
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
    """Strategia: Modello lineare con bias. adj_val = pred_val * scale + bias"""
    nutrients = strategy_cfg["nutrients"]
    n_trials = strategy_cfg["n_trials"]
    best_params = {}
    total_mae_perc = 0.0

    for nut in nutrients:
        def objective(trial, _nut=nut):
            scale = trial.suggest_float(f"scale_{_nut}", 0.005, 5.0)

            # Impostazione dinamica dei range di bias in base al nutriente
            if _nut in ["mass", "cal"]:
                bias_limit = 500.0  # Range ampio per grammi totali e calorie
            else:
                bias_limit = 150.0  # Range ridotto per fat, carbs, protein

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

                # Applica la trasformazione lineare (mx + q)
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


# Mappa strategia -> funzione di ottimizzazione
STRATEGY_FUNCTIONS = {
    "linear_all5": optimize_linear_independent,
    "linear_mass_only": optimize_linear_independent,
    "power_law_all5": optimize_power_law_all5,
    "piecewise_all5": optimize_piecewise_all5,
    "linear_bias_all5": optimize_linear_bias_all5,
}


def optimize_for_file(predictions_file, strategy_name):
    """Entry point: ottimizza un file di predizioni con la strategia indicata."""
    if strategy_name not in STRATEGY_CATALOG:
        raise ValueError(f"Strategia '{strategy_name}' non trovata. "
                         f"Disponibili: {list(STRATEGY_CATALOG.keys())}")
    strategy_cfg = STRATEGY_CATALOG[strategy_name]
    opt_fn = STRATEGY_FUNCTIONS[strategy_name]
    predictions_data = load_json(predictions_file)
    print(f"   Strategia: {strategy_name} | Nutrienti: {strategy_cfg['nutrients']} | "
          f"Trials: {strategy_cfg['n_trials']}")
    return opt_fn(predictions_data, strategy_cfg)


def run_strategy(strategy_name, json_files):
    """
    Esegue una strategia completa su tutti i json_files e salva il summary.
    Ogni studio Optuna usa n_jobs=-1 per sfruttare tutti i core disponibili.
    Salta l'ottimizzazione se i risultati sono già presenti nel summary.
    """
    strategy_cfg = STRATEGY_CATALOG[strategy_name]
    folder_results = {}

    prefix = f"[{strategy_name}]"
    print(f"\n{prefix} ▶ Avvio | {strategy_cfg['description']} | "
          f"Nutrienti: {strategy_cfg['nutrients']} | Trials: {strategy_cfg['n_trials']}")

    for filepath in sorted(json_files):
        config_name = os.path.basename(filepath)
        folder_path = os.path.dirname(filepath)
        strategy_folder = os.path.join(folder_path, strategy_name)
        summary_file = os.path.join(strategy_folder, "optuna_summary.json")

        # --- NUOVA LOGICA DI SKIP ---
        already_done = False
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

                # Se il file corrente è già stato calcolato in precedenza, salta
                if config_name in existing_data:
                    already_done = True
                    print(f"{prefix} -> Ottimizzazione saltata (già calcolata): {config_name}")

                    # Conserva il dato pre-esistente per non perderlo in fase di salvataggio finale
                    if strategy_folder not in folder_results:
                        folder_results[strategy_folder] = {}
                    folder_results[strategy_folder][config_name] = existing_data[config_name]
            except Exception:
                pass

        if already_done:
            continue
        # ----------------------------

        print(f"{prefix} -> Ottimizzazione: {config_name}")
        try:
            best_mae, best_params = optimize_for_file(filepath, strategy_name)
            print(f"{prefix}    MAE% Medio: {best_mae:.2f}%")
        except Exception as e:
            print(f"{prefix} [!] Errore su {filepath}: {e}")
            continue

        os.makedirs(strategy_folder, exist_ok=True)

        if strategy_folder not in folder_results:
            folder_results[strategy_folder] = {}
        folder_results[strategy_folder][config_name] = {
            "best_combined_mae_perc": round(best_mae, 2),
            "optimal_scalars": best_params
        }

    # Salvataggio JSON con _strategy_info in testa
    strategy_info_block = {
        "name": strategy_name,
        "description": strategy_cfg["description"],
        "nutrients_optimized": strategy_cfg["nutrients"],
        "param_type": strategy_cfg["param_type"],
        "n_trials": strategy_cfg["n_trials"],
    }

    for strategy_folder, results in folder_results.items():
        json_filename = os.path.join(strategy_folder, "optuna_summary.json")

        # Unisce i dati nuovi con quelli vecchi se il summary esiste già
        final_output = {"_strategy_info": strategy_info_block}
        if os.path.exists(json_filename):
            try:
                with open(json_filename, 'r', encoding='utf-8') as f:
                    old_data = json.load(f)
                    final_output.update(old_data)
            except Exception:
                pass

        # Sovrascrive/aggiorna le chiavi elaborate in questo ciclo
        final_output.update(results)

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4)
        print(f"[{strategy_name}] ✅ Salvato: {json_filename}")

    return strategy_name, folder_results

if __name__ == "__main__":
    # RUN_MODE controlla quali cartelle elaborare:
    #   "both"   (default) -> RAG + non-RAG
    #   "rag"              -> solo cartelle *_rag
    #   "nonrag"           -> solo cartelle senza suffisso _rag
    RUN_MODE = os.environ.get("RUN_MODE", "both").lower()
    if RUN_MODE not in ("both", "rag", "nonrag"):
        print(f"[ERRORE] RUN_MODE='{RUN_MODE}' non valido. Scegli tra: both, rag, nonrag")
        exit(1)

    # Consente di lanciare una sola strategia via env var.
    OPTUNA_STRATEGY_ENV = os.environ.get("OPTUNA_STRATEGY", "all")

    # MODIFICA: Ricerca in tutte le cartelle principali (es. few_shot_standard, few_shot_yolo)
    # Aggiunto un "*" per attraversare il livello delle cartelle principali
    pattern_rag    = os.path.join(PREDICTIONS_BASE_DIR, "*", "*_rag", "*.json")
    pattern_nonrag = os.path.join(PREDICTIONS_BASE_DIR, "*", "*", "*.json")   # cattura tutto

    if RUN_MODE == "rag":
        all_json_files = glob.glob(pattern_rag)
        print("Modalità: solo cartelle RAG (*_rag)")
    elif RUN_MODE == "nonrag":
        # Prende tutte le cartelle, poi esclude quelle che terminano con _rag
        all_json_files = [
            f for f in glob.glob(pattern_nonrag)
            if not os.path.basename(os.path.dirname(f)).endswith("_rag")
        ]
        print("Modalità: solo cartelle non-RAG")
    else:  # both
        all_json_files = glob.glob(pattern_nonrag)
        print("Modalità: tutte le cartelle (RAG + non-RAG)")

    # Esclude: optuna_summary.json e file dentro sottocartelle di strategia
    strategy_names_set = set(STRATEGY_CATALOG.keys())
    json_files = [
        f for f in all_json_files
        if os.path.basename(f) != "optuna_summary.json"
        and not any(part in strategy_names_set for part in f.replace(os.sep, '/').split('/'))
    ]

    if not json_files:
        print("Nessun file JSON trovato. Controlla il percorso o il pattern di ricerca.")
        exit(0)

    # Selezione strategie da eseguire
    if OPTUNA_STRATEGY_ENV == "all" or OPTUNA_STRATEGY_ENV not in STRATEGY_CATALOG:
        strategies_to_run = list(STRATEGY_CATALOG.keys())
    else:
        strategies_to_run = [OPTUNA_STRATEGY_ENV]

    n_cores = os.cpu_count()
    print(f"\n{'=' * 60}")
    print(f"Strategie da eseguire ({len(strategies_to_run)}): {strategies_to_run}")
    print(f"File di predizione trovati: {len(json_files)}")
    print(f"Core disponibili: {n_cores} (usati da ogni studio Optuna via n_jobs=-1)")
    print(f"{'=' * 60}\n")

    # Esecuzione sequenziale delle strategie: ogni studio usa tutti i core
    all_results = []
    for strategy_name in strategies_to_run:
        result = run_strategy(strategy_name, json_files)
        all_results.append(result)

    print(f"\n{'=' * 60}")
    print("✅ Ottimizzazione batch completata per tutte le strategie!")
    for strategy_name, _ in all_results:
        print(f"   • {strategy_name}")
    print(f"{'=' * 60}")
