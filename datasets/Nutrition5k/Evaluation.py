import json
import os
import csv
import statistics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_vocabulary(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File CSV non trovato: {path}")

    vocab = set()
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ing = row.get('ingr', '').strip().lower()
            if ing:
                vocab.add(ing)
    return list(vocab)


class Nutrition5kEvaluator:
    def __init__(self, pred_json_path: str, nlp_model, vocabulary: list, vocab_embeddings, threshold: float):
        self.pred_path = pred_json_path
        self.threshold = threshold
        self.nlp_model = nlp_model
        self.vocabulary = vocabulary
        self.vocab_embeddings = vocab_embeddings

        self.predictions = self._load_json(self.pred_path)
        self.similarity_log = []

    def _load_json(self, path: str) -> list:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File JSON non trovato: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _clean_ingredients(self, ingredients: list) -> list:
        return [str(ing.get("name", "")).strip().lower()
                for ing in ingredients
                if isinstance(ing, dict) and str(ing.get("name", "")).strip()]

    def _standardize_list(self, pred_list: list, recipe_id: str = "N/A") -> set:
        if not pred_list:
            return set()

        pred_embeddings = self.nlp_model.encode(pred_list)
        sim_matrix = cosine_similarity(pred_embeddings, self.vocab_embeddings)
        standardized_preds = set()

        for i, pred_ing in enumerate(pred_list):
            best_match_idx = np.argmax(sim_matrix[i])
            best_score = float(sim_matrix[i][best_match_idx])
            accepted = best_score >= self.threshold
            best_vocab_match = self.vocabulary[best_match_idx]

            self.similarity_log.append({
                "recipe_id": recipe_id,
                "predicted": pred_ing,
                "best_vocab_match": best_vocab_match,
                "max_similarity_score": round(best_score, 4),
                "threshold_passed": accepted
            })

            if accepted:
                standardized_preds.add(best_vocab_match)
            else:
                standardized_preds.add(pred_ing)

        return standardized_preds

    def _compute_set_metrics(self, pred_set: set, gt_set: set) -> dict:
        intersection = len(gt_set.intersection(pred_set))
        union = len(gt_set.union(pred_set))

        precision = intersection / len(pred_set) if pred_set else 0.0
        recall = intersection / len(gt_set) if gt_set else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = intersection / union if union > 0 else 0.0

        return {"P": precision, "R": recall, "F1": f1, "IoU": iou}

    def evaluate_nlp_metrics(self) -> dict:
        sums = {
            "non_std": {"P": 0.0, "R": 0.0, "F1": 0.0, "IoU": 0.0},
            "std": {"P": 0.0, "R": 0.0, "F1": 0.0, "IoU": 0.0}
        }
        valid_items = 0
        self.similarity_log = []

        for item in self.predictions:
            gt_raw = item.get("ground_truth_ingredients", [])
            pred_dict = item.get("predicted_nutrition", {})
            pred_raw = pred_dict.get("visible_ingredients", []) + pred_dict.get("hidden_ingredients", [])

            recipe_id = item.get("food_id", "unknown")
            gt_list = self._clean_ingredients(gt_raw)
            pred_list = self._clean_ingredients(pred_raw)

            if not gt_list:
                continue

            gt_set = set(gt_list)
            pred_set_exact = set(pred_list)

            m_exact = self._compute_set_metrics(pred_set_exact, gt_set)
            for k in sums["non_std"]: sums["non_std"][k] += m_exact[k]

            pred_set_std = self._standardize_list(pred_list, recipe_id)
            m_std = self._compute_set_metrics(pred_set_std, gt_set)
            for k in sums["std"]: sums["std"][k] += m_std[k]

            valid_items += 1

        if valid_items == 0:
            return {}

        return {
            "Non-Standardized": {k: v / valid_items for k, v in sums["non_std"].items()},
            "Standardized": {k: v / valid_items for k, v in sums["std"].items()}
        }

    def evaluate_macronutrients(self) -> dict:
        DATA_FIELDNAMES = ["calories", "mass", "fat", "carbohydrates", "protein"]
        groundtruth_values = {field: [] for field in DATA_FIELDNAMES}
        err_values = {field: [] for field in DATA_FIELDNAMES}

        for item in self.predictions:
            gt_cal = gt_mass = gt_fat = gt_carbs = gt_prot = 0.0
            for ing in item.get("ground_truth_ingredients", []):
                gt_mass += ing.get("weight_grams", 0.0)
                macros = ing.get("macronutrients", {})
                gt_cal += macros.get("calories", 0.0)
                gt_fat += macros.get("fat", 0.0)
                gt_carbs += macros.get("carbohydrates", 0.0)
                gt_prot += macros.get("protein", 0.0)

            pred_nut = item.get("predicted_nutrition", {})
            pred_macros = pred_nut.get("total_macronutrients", {})

            pred_cal = pred_macros.get("calories", 0.0)
            pred_fat = pred_macros.get("fat", 0.0)
            pred_carbs = pred_macros.get("carbohydrates", 0.0)
            pred_prot = pred_macros.get("protein", 0.0)

            pred_mass = 0.0
            for ing in pred_nut.get("visible_ingredients", []) + pred_nut.get("hidden_ingredients", []):
                pred_mass += ing.get("weight_grams", 0.0)

            gt_vals = {"calories": gt_cal, "mass": gt_mass, "fat": gt_fat, "carbohydrates": gt_carbs,
                       "protein": gt_prot}
            pred_vals = {"calories": pred_cal, "mass": pred_mass, "fat": pred_fat, "carbohydrates": pred_carbs,
                         "protein": pred_prot}

            for field in DATA_FIELDNAMES:
                groundtruth_values[field].append(gt_vals[field])
                err_values[field].append(abs(pred_vals[field] - gt_vals[field]))

        output_stats = {}
        for field in DATA_FIELDNAMES:
            mean_err = statistics.mean(err_values[field])
            mean_gt = statistics.mean(groundtruth_values[field])
            output_stats[f"{field}_MAE"] = mean_err
            output_stats[f"{field}_MAE_%"] = (100 * mean_err / mean_gt) if mean_gt > 0 else 0.0

        return output_stats


# ==========================================
# FUNZIONI PER I GRAFICI
# ==========================================

def plot_nlp_metrics(all_nlp_results: dict, show_standardized: bool, output_dir: str):
    metrics = ["P", "R", "F1", "IoU"]
    models = list(all_nlp_results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(models)}

    # --- PLOT COMBINATO (SUBPLOTS) ---
    fig_combined, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, m in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(models))

        if show_standardized:
            width = 0.35
            non_std_vals = [all_nlp_results[mod]["Non-Standardized"][m] for mod in models]
            std_vals = [all_nlp_results[mod]["Standardized"][m] for mod in models]

            for i, mod in enumerate(models):
                ax.bar(x[i] - width / 2, non_std_vals[i], width, color=model_colors[mod], edgecolor='black')
                ax.bar(x[i] + width / 2, std_vals[i], width, color=model_colors[mod], alpha=0.5, hatch='//',
                       edgecolor='black')
        else:
            width = 0.5
            non_std_vals = [all_nlp_results[mod]["Non-Standardized"][m] for mod in models]
            for i, mod in enumerate(models):
                ax.bar(x[i], non_std_vals[i], width, color=model_colors[mod], edgecolor='black')

        ax.set_ylabel('Score')
        ax.set_title(f'Metrica NLP: {m}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.grid(False)
        ax.set_ylim(0, 1.05)

    legend_elements = []
    for mod in models:
        if show_standardized:
            legend_elements.append(Patch(facecolor=model_colors[mod], edgecolor='black', label=f"{mod} (Non-Std)"))
            legend_elements.append(
                Patch(facecolor=model_colors[mod], alpha=0.5, hatch='//', edgecolor='black', label=f"{mod} (Std)"))
        else:
            legend_elements.append(Patch(facecolor=model_colors[mod], edgecolor='black', label=mod))

    fig_combined.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.suptitle("Confronto Riepilogativo delle Metriche NLP", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    comb_path = os.path.join(output_dir, 'nlp_metrics_combined.png')
    plt.savefig(comb_path, bbox_inches='tight', dpi=300)
    plt.close()

    # --- PLOT SINGOLI ---
    for m in metrics:
        fig_single, ax_single = plt.subplots(figsize=(8, 6))
        x = np.arange(len(models))

        if show_standardized:
            width = 0.35
            non_std_vals = [all_nlp_results[mod]["Non-Standardized"][m] for mod in models]
            std_vals = [all_nlp_results[mod]["Standardized"][m] for mod in models]

            for i, mod in enumerate(models):
                ax_single.bar(x[i] - width / 2, non_std_vals[i], width, color=model_colors[mod], edgecolor='black')
                ax_single.bar(x[i] + width / 2, std_vals[i], width, color=model_colors[mod], alpha=0.5, hatch='//',
                              edgecolor='black')
        else:
            width = 0.5
            non_std_vals = [all_nlp_results[mod]["Non-Standardized"][m] for mod in models]
            for i, mod in enumerate(models):
                ax_single.bar(x[i], non_std_vals[i], width, color=model_colors[mod], edgecolor='black')

        ax_single.set_ylabel('Score')
        ax_single.set_title(f'Metrica NLP: {m}', fontweight='bold')
        ax_single.set_xticks(x)
        ax_single.set_xticklabels(models, rotation=15, ha="right")
        ax_single.grid(False)
        ax_single.set_ylim(0, 1.05)

        ax_single.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
        plt.tight_layout()
        single_path = os.path.join(output_dir, f'nlp_metric_{m}.png')
        plt.savefig(single_path, bbox_inches='tight', dpi=300)
        plt.close()


def plot_macro_metrics(all_macro_results: dict, show_mae_percentage: bool, output_dir: str):
    fields = ["calories", "mass", "fat", "carbohydrates", "protein"]

    # Creazione dei dati per il plot includendo i due modelli "baseline" presi dalla Tabella 3 del paper
    plot_data = dict(all_macro_results)

    plot_data["Paper 2D Direct"] = {
        "calories_MAE": 70.6, "calories_MAE_%": 26.1,
        "mass_MAE": 40.4, "mass_MAE_%": 18.8,
        "fat_MAE": 5.0, "fat_MAE_%": 34.2,
        "carbohydrates_MAE": 6.1, "carbohydrates_MAE_%": 31.9,
        "protein_MAE": 5.5, "protein_MAE_%": 29.5
    }

    plot_data["Paper Baseline (Std)"] = {
        "calories_MAE": 150.8, "calories_MAE_%": 60.2,
        "mass_MAE": 124.6, "mass_MAE_%": 58.5,
        "fat_MAE": 8.2, "fat_MAE_%": 67.6,
        "carbohydrates_MAE": 12.5, "carbohydrates_MAE_%": 62.1,
        "protein_MAE": 10.5, "protein_MAE_%": 62.1
    }

    models = list(plot_data.keys())

    # Colori: 4 per gemma, e 2 dedicati per le baseline del paper
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(models)}

    # Preparazione legenda combinata/singola
    legend_elements = []
    for mod in models:
        if show_mae_percentage:
            legend_elements.append(Patch(facecolor=model_colors[mod], edgecolor='black', label=f"{mod} (Abs)"))
            legend_elements.append(
                Patch(facecolor=model_colors[mod], alpha=0.5, hatch='//', edgecolor='black', label=f"{mod} (%)"))
        else:
            legend_elements.append(Patch(facecolor=model_colors[mod], edgecolor='black', label=mod))

    # --- PLOT COMBINATO (SUBPLOTS) ---
    fig_combined, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, f in enumerate(fields):
        ax = axes[idx]
        x = np.arange(len(models))

        if show_mae_percentage:
            width = 0.35
            for i, mod in enumerate(models):
                # Colonna MAE Assoluto (Piena)
                ax.bar(x[i] - width / 2, plot_data[mod][f"{f}_MAE"], width, color=model_colors[mod], edgecolor='black',
                       zorder=2)
                # Colonna MAE % (Tratteggiata, vicina a quella assoluta)
                ax.bar(x[i] + width / 2, plot_data[mod][f"{f}_MAE_%"], width, color=model_colors[mod], alpha=0.5,
                       hatch='//', edgecolor='black', zorder=2)
        else:
            width = 0.5
            for i, mod in enumerate(models):
                ax.bar(x[i], plot_data[mod][f"{f}_MAE"], width, color=model_colors[mod], edgecolor='black', zorder=2)

        ax.set_ylabel('Errore Medio Assoluto (Valore e %)')
        ax.set_title(f'MAE & MAE%: {f.capitalize()}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.grid(False)

    axes[5].axis('off')

    # Aggiusta il numero di colonne nella legenda in base a quanti elementi ci sono
    ncols_legend = 4 if show_mae_percentage else 3
    fig_combined.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=ncols_legend)

    plt.suptitle("Confronto Errore Medio Assoluto (MAE) dei Macronutrienti", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    comb_path = os.path.join(output_dir, 'macro_mae_combined.png')
    plt.savefig(comb_path, bbox_inches='tight', dpi=300)
    plt.close()

    # --- PLOT SINGOLI ---
    for f in fields:
        fig_single, ax_single = plt.subplots(figsize=(10, 8))
        x = np.arange(len(models))

        if show_mae_percentage:
            width = 0.35
            for i, mod in enumerate(models):
                ax_single.bar(x[i] - width / 2, plot_data[mod][f"{f}_MAE"], width, color=model_colors[mod],
                              edgecolor='black', zorder=2)
                ax_single.bar(x[i] + width / 2, plot_data[mod][f"{f}_MAE_%"], width, color=model_colors[mod], alpha=0.5,
                              hatch='//', edgecolor='black', zorder=2)
        else:
            width = 0.5
            for i, mod in enumerate(models):
                ax_single.bar(x[i], plot_data[mod][f"{f}_MAE"], width, color=model_colors[mod], edgecolor='black',
                              zorder=2)

        ax_single.set_ylabel('Errore Medio Assoluto (Valore e %)')
        ax_single.set_title(f'MAE & MAE%: {f.capitalize()}', fontweight='bold')
        ax_single.set_xticks(x)
        ax_single.set_xticklabels(models, rotation=25, ha="right")
        ax_single.grid(False)

        ax_single.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=ncols_legend)

        plt.tight_layout()
        single_path = os.path.join(output_dir, f'macro_mae_{f}.png')
        plt.savefig(single_path, bbox_inches='tight', dpi=300)
        plt.close()


if __name__ == "__main__":

    # ================= CONFIGURAZIONI =================
    THRESHOLD = 0.7
    SHOW_STANDARDIZED_METRICS = True  # Attiva/Disattiva NLP Standardizzate (IoU, F1...)
    SHOW_MAE_PERCENTAGE = True  # Attiva/Disattiva plotting del MAE% affiancato al MAE
    OUTPUT_DIR = "data/evaluation"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    PRED_JSON_PATHS = [
        "data/prediction/test_predictions_nutrition5k_gemma4e2b_q4.json",
        "data/prediction/test_predictions_nutrition5k_gemma4e2b_q8.json",
        "data/prediction/test_predictions_nutrition5k_gemma_e2b_q8V2.json",
        "data/prediction/test_predictions_nutrition5k_gemma4e4b_q4.json",
        "data/prediction/test_predictions_nutrition5k_gemma4e4b_q8.json",
        "data/prediction/test_predictions_nutrition5k_gemma4_e4b_q8V2.json"
    ]
    INGREDIENTS_CSV_PATH = "data/metadata/ingredients_metadata.csv"
    # ==================================================

    print("Caricamento del vocabolario da CSV...")
    vocabulary = load_vocabulary(INGREDIENTS_CSV_PATH)

    print("Caricamento del modello NLP: all-MiniLM-L6-v2... (Avviene una singola volta)")
    nlp_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Calcolo degli embedding per il vocabolario base...")
    vocab_embeddings = nlp_model.encode(vocabulary)

    all_nlp_results = {}
    all_macro_results = {}

    for pred_path in PRED_JSON_PATHS:
        model_name = os.path.basename(pred_path).replace('test_predictions_nutrition5k_', '').replace('.json', '')

        print(f"\n{'=' * 70}")
        print(f"VALUTAZIONE MODELLO: {model_name}")
        print(f"{'=' * 70}")

        try:
            evaluator = Nutrition5kEvaluator(
                pred_json_path=pred_path,
                nlp_model=nlp_model,
                vocabulary=vocabulary,
                vocab_embeddings=vocab_embeddings,
                threshold=THRESHOLD
            )

            nlp_res = evaluator.evaluate_nlp_metrics()
            all_nlp_results[model_name] = nlp_res

            macro_res = evaluator.evaluate_macronutrients()
            all_macro_results[model_name] = macro_res

        except Exception as e:
            print(f"\n[!] Errore durante l'esecuzione del file {pred_path}:\n{e}")

    print(f"\n{'=' * 70}")
    print("VALUTAZIONE COMPLETATA. GENERAZIONE GRAFICI IN CORSO...")
    print(f"{'=' * 70}")

    if all_nlp_results:
        plot_nlp_metrics(all_nlp_results, show_standardized=SHOW_STANDARDIZED_METRICS, output_dir=OUTPUT_DIR)
    if all_macro_results:
        # Passiamo la nuova variabile SHOW_MAE_PERCENTAGE
        plot_macro_metrics(all_macro_results, show_mae_percentage=SHOW_MAE_PERCENTAGE, output_dir=OUTPUT_DIR)

    print(f"\nTutti i grafici (singoli e combinati) sono stati salvati nella cartella: {OUTPUT_DIR}")