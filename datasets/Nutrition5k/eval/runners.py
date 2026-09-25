"""
runners.py
==========
Orchestrazione delle valutazioni: legge le predizioni, calcola le metriche
(tramite Nutrition5kEvaluator), gestisce la cache su CSV/JSON e richiama i
grafici in plots.py. Contiene SOLO le pipeline che alimentano grafici
few-shot (nessuna valutazione "singolo modello a barre").
"""
import os
import csv
import json

from . import style as st
from .evaluator import Nutrition5kEvaluator
from .data_utils import parse_model_name
from . import plots


# ---------------------------------------------------------------------------
# 1) Few-shot learning (NLP + Macro), 1/5/10 esempi
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

    fs_csv_path = os.path.join("data/evaluation", f"few_shot_evaluation{csv_suffix}.csv")
    fs_yolo_csv_path = os.path.join("data/evaluation", f"few_shot_yolo_evaluation{csv_suffix}_{optuna_strategy}.csv")

    def load_fs_csv(path):
        if not os.path.exists(path):
            return False
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                g, v = row['Group'], row['Variant']
                if g in all_results and v in all_results[g]:
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
    loaded_fs_yolo = load_fs_csv(fs_yolo_csv_path)
    updated_fs = updated_fs_yolo = False

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.endswith('.json'):
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
            var_name_optuna = f"{prefix} (YOLO Optuna)" if is_yolo else None

            needs_base = not (loaded_fs and not is_yolo and all_results[g].get(var_name) is not None) and \
                         not (loaded_fs_yolo and is_yolo and all_results[g].get(var_name) is not None)
            needs_optuna = is_yolo and not (loaded_fs_yolo and all_results[g].get(var_name_optuna) is not None)

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
                    updated_fs_yolo = updated_fs_yolo or is_yolo
                    updated_fs = updated_fs or (not is_yolo)

                if needs_optuna:
                    print(f"Evaluating Few-Shot: {var_name_optuna} - {g}")
                    optuna_summary_path = os.path.join("data", "optuna", "few_shot_yolo", folder,
                                                         optuna_strategy, "optuna_summary.json")
                    optuna_scalars = None
                    if os.path.exists(optuna_summary_path):
                        with open(optuna_summary_path, 'r', encoding='utf-8') as f:
                            optuna_data = json.load(f)
                        model_identifier = filename.replace('test_predictions_nutrition5k_yolo_', '').replace('.json', '')
                        for key, val in optuna_data.items():
                            if model_identifier in key:
                                optuna_scalars = val.get("optimal_scalars")
                                break
                    if optuna_scalars:
                        evaluator_opt = Nutrition5kEvaluator(pred_path, resources.nlp_model, resources.vocabulary, resources.vocab_embeddings,
                                                               threshold, optuna_scalars=optuna_scalars)
                        all_results[g][var_name_optuna] = {
                            "nlp": evaluator_opt.evaluate_nlp_metrics(),
                            "macro": evaluator_opt.evaluate_macronutrients(),
                        }
                        updated_fs_yolo = True
                    else:
                        print(f"[!] Could not find optuna scalars for {filename} in {optuna_summary_path}")
            except Exception as e:
                print(f"[!] Error in {pred_path}: {e}")

    def save_fs_csv(path, is_yolo):
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
                for var in few_shot_variants:
                    if ("YOLO" in var) != is_yolo:
                        continue
                    res = all_results[grp][var]
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
        save_fs_csv(fs_csv_path, False)
    if updated_fs_yolo:
        save_fs_csv(fs_yolo_csv_path, True)

    # --- Grafici (solo stile few-shot) ---
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

    return all_results


# ---------------------------------------------------------------------------
# 2) Confronto strategie Optuna (dot-plot, stile few-shot)
# ---------------------------------------------------------------------------
def evaluate_optuna_strategy_comparison(resources, threshold,
                                         rag=False, strategies=None, base_dir="data/prediction"):
    strategies = strategies or st.OPTUNA_STRATEGIES
    run_label = "RAG" if rag else "non-RAG"
    print(f"\n{'=' * 70}\nOPTUNA STRATEGY COMPARISON ({run_label})\nStrategie: {strategies}\n{'=' * 70}")

    if rag:
        folders = ["one_example_rag", "five_example_rag", "ten_example_rag"]
        folder_to_prefix = {"one_example_rag": "1 Example", "five_example_rag": "5 Examples",
                             "ten_example_rag": "10 Examples"}
        cache_path = "data/evaluation/optuna_strategy_comparison_rag.json"
        output_dir = "data/evaluation/optuna_strategy_metrics_rag"
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
            print(f"Cache caricata da {cache_path}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[!] Cache corrotta ({e}), verrà ricreata.")
            results = {g: {} for g in st.GROUPS}

    updated = False

    for folder in folders:
        prefix = folder_to_prefix[folder]
        pred_folder = os.path.join(base_dir, folder)
        if not os.path.exists(pred_folder):
            print(f"[!] Cartella predizioni non trovata: {pred_folder}")
            continue

        # Cache locale dei summary Optuna per questo folder (caricati una volta per strategia)
        _optuna_cache = {}

        for filename in sorted(os.listdir(pred_folder)):
            if not filename.endswith('.json') or 'yolo' not in filename.lower():
                continue
            g, _ = parse_model_name(filename)
            if g not in st.GROUPS:
                continue

            pred_path = os.path.join(pred_folder, filename)
            model_identifier = filename.replace('test_predictions_nutrition5k_yolo_', '').replace('.json', '')

            results.setdefault(g, {})
            yolo_base_var = f"{prefix} ({st.YOLO_BASE_KEY})"
            if results[g].get(yolo_base_var) is None:
                print(f"  Evaluating [YOLO base] {yolo_base_var} - {g}")
                try:
                    ev_base = Nutrition5kEvaluator(pred_path, resources.nlp_model, resources.vocabulary,
                                                   resources.vocab_embeddings, threshold)
                    results[g][yolo_base_var] = {"macro": ev_base.evaluate_macronutrients()}
                    updated = True
                except Exception as e:
                    print(f"[!] Errore YOLO base su {pred_path}: {e}")

            for strategy in strategies:
                var_name = f"{prefix} ({strategy})"
                if results.get(g, {}).get(var_name) is not None:
                    continue

                optuna_summary_path = os.path.join("data", "optuna", "few_shot_yolo", folder, strategy,
                                                     "optuna_summary.json")
                if not os.path.exists(optuna_summary_path):
                    print(f"[!] Optuna summary non trovato: {optuna_summary_path}")
                    continue

                if optuna_summary_path not in _optuna_cache:
                    try:
                        with open(optuna_summary_path, 'r', encoding='utf-8') as f:
                            _optuna_cache[optuna_summary_path] = json.load(f)
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"[!] Errore lettura {optuna_summary_path}: {e}")
                        _optuna_cache[optuna_summary_path] = {}
                optuna_data = _optuna_cache[optuna_summary_path]

                optuna_scalars = None
                for key, val in optuna_data.items():
                    if key.startswith("_"):
                        continue
                    if model_identifier in key:
                        optuna_scalars = val.get("optimal_scalars")
                        break
                if not optuna_scalars:
                    print(f"[!] Scalars non trovati per {filename} in {optuna_summary_path}")
                    continue

                print(f"  Evaluating [{strategy}] {var_name} - {g}")
                try:
                    evaluator = Nutrition5kEvaluator(pred_path, resources.nlp_model, resources.vocabulary,
                                                     resources.vocab_embeddings, threshold,
                                                     optuna_scalars=optuna_scalars)
                    results[g][var_name] = {"macro": evaluator.evaluate_macronutrients()}
                    updated = True
                except Exception as e:
                    print(f"[!] Errore su {pred_path} con {strategy}: {e}")

    if updated:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"Cache salvata in {cache_path}")

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

    print(f"Grafici Optuna Strategy salvati in {output_dir}")

    from . import tables
    tables.print_optuna_strategy_tables(results, strategies)

    return results


# ---------------------------------------------------------------------------
# 3) Confronto YOLO vs YOLO Optuna (dot-plot, stile few-shot)
# ---------------------------------------------------------------------------
def evaluate_yolo_optuna_comparison(resources, threshold,
                                     base_dir="data/prediction", rag=False, optuna_strategy="linear_all5"):
    run_label = "RAG" if rag else "Normale"
    print(f"\n{'=' * 70}\nSTARTING YOLO OPTUNA EVALUATION COMPARISON ({run_label}, strategy={optuna_strategy})\n{'=' * 70}")

    if rag:
        folders = ["one_example_rag", "five_example_rag", "ten_example_rag"]
        folder_to_prefix = {"one_example_rag": "1 Example", "five_example_rag": "5 Examples",
                             "ten_example_rag": "10 Examples"}
        yolo_optuna_json_path = f"data/evaluation/few_shot_yolo_optuna_evaluation_rag_{optuna_strategy}.json"
        output_dir = f"data/evaluation/yolo_optuna_metrics_rag_{optuna_strategy}"
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
            print(f"[!] Cache file corrotto o vuoto ({e}), verrà ricreato da zero.")
            os.remove(yolo_optuna_json_path)
            results = {}

    updated = False

    for folder in folders:
        pred_folder = os.path.join(base_dir, folder)
        optuna_summary_path = os.path.join("data", "optuna", "few_shot_yolo", folder,
                                             optuna_strategy, "optuna_summary.json")
        if not os.path.exists(pred_folder) or not os.path.exists(optuna_summary_path):
            print(f"[!] Skipping {folder}: pred_folder o optuna_summary non trovati (strategy='{optuna_strategy}')")
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
# 4) Studio soglia (loop di threshold su tutti i modelli one-example)
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
# 5) Valutazione per-modello (one_example): alimenta il plot delle metriche
#    NLP (bar chart) e le tabelle riassuntive. Usa la stessa cache CSV
#    dell'originale (nlp_evaluation.csv): se un modello è già presente non
#    viene ricalcolato, quindi eventualmente il modello NLP non viene mai
#    caricato.
# ---------------------------------------------------------------------------
def evaluate_single_run_models(resources, threshold, pred_json_paths, eval_csv_path, similarity_dir=None):
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
        model_name_optuna = model_name + "_optuna"

        needs_base = model_name not in all_nlp_results or model_name not in all_macro_results
        needs_optuna = is_yolo and (model_name_optuna not in all_nlp_results or model_name_optuna not in all_macro_results)

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
                print(f"--- Evaluating YOLO Optuna variant: {model_name_optuna} ---")
                optuna_summary_path = os.path.join("data", "optuna", "few_shot_yolo", "one_example",
                                                     "optuna_summary.json")
                optuna_scalars = None
                if os.path.exists(optuna_summary_path):
                    with open(optuna_summary_path, 'r', encoding='utf-8') as f:
                        optuna_data = json.load(f)
                    model_identifier = model_name.replace("yolo_", "")
                    for key, val in optuna_data.items():
                        if model_identifier in key:
                            optuna_scalars = val.get("optimal_scalars")
                            break

                if optuna_scalars:
                    evaluator_opt = Nutrition5kEvaluator(pred_path, resources.nlp_model, resources.vocabulary,
                                                          resources.vocab_embeddings, threshold,
                                                          optuna_scalars=optuna_scalars)
                    all_nlp_results[model_name_optuna] = evaluator_opt.evaluate_nlp_metrics()
                    all_macro_results[model_name_optuna] = evaluator_opt.evaluate_macronutrients()
                else:
                    print(f"[!] Could not find optuna scalars for {model_name} in {optuna_summary_path}")
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
    strategies = ['Standard', 'YOLO', 'YOLO Optuna']

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