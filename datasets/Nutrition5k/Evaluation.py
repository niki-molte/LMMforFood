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

            # --- MODIFICA: Auto-rilevamento della pipeline ---
            if "predicted_nutrition_step2" in item:
                pred_dict = item.get("predicted_nutrition_step2", {})
            else:
                pred_dict = item.get("predicted_nutrition", {})
            # -------------------------------------------------

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
            # --- GROUND TRUTH ---
            gt_cal = gt_mass = gt_fat = gt_carbs = gt_prot = 0.0
            for ing in item.get("ground_truth_ingredients", []):
                gt_mass += ing.get("weight_grams", 0.0)
                macros = ing.get("macronutrients", {})
                gt_cal += macros.get("calories", 0.0)
                gt_fat += macros.get("fat", 0.0)
                gt_carbs += macros.get("carbohydrates", 0.0)
                gt_prot += macros.get("protein", 0.0)

            # --- PREDICTIONS ---
            # --- MODIFICA: Auto-rilevamento della pipeline ---
            if "predicted_nutrition_step2" in item:
                pred_nut = item.get("predicted_nutrition_step2", {})
            else:
                pred_nut = item.get("predicted_nutrition", {})
            # -------------------------------------------------

            pred_cal = pred_mass = pred_fat = pred_carbs = pred_prot = 0.0
            predicted_ingredients = pred_nut.get("visible_ingredients", []) + pred_nut.get("hidden_ingredients", [])

            for ing in predicted_ingredients:
                pred_mass += ing.get("weight_grams", 0.0)

                macros = ing.get("macronutrients", {})
                if isinstance(macros, dict):
                    pred_cal += macros.get("calories", 0.0)
                    pred_fat += macros.get("fat", 0.0)
                    pred_carbs += macros.get("carbohydrates", 0.0)
                    pred_prot += macros.get("protein", 0.0)

            # --- COMPILAZIONE RISULTATI ---
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

GROUPS = ['E2B - Q4', 'E2B - Q8', 'E4B - Q4', 'E4B - Q8']
VARIANTS = ['Standard (Alta Temp.)', 'Standard (Bassa Temp.)', 'Two-Step (Bassa Temp.)']
VARIANT_COLORS = {
    'Standard (Alta Temp.)': '#404040',        # Grigio scuro
    'Standard (Bassa Temp.)': '#1f77b4',       # Blu standard
    'Two-Step (Bassa Temp.)': '#87ceeb'        # Azzurro chiaro
}

def parse_model_name(name: str):
    name_lower = name.lower()
    group = "Unknown"
    if "e2b" in name_lower and "q4" in name_lower:
        group = "E2B - Q4"
    elif "e2b" in name_lower and "q8" in name_lower:
        group = "E2B - Q8"
    elif "e4b" in name_lower and "q4" in name_lower:
        group = "E4B - Q4"
    elif "e4b" in name_lower and "q8" in name_lower:
        group = "E4B - Q8"
        
    variant = "Unknown"
    if "twostep" in name_lower:
        variant = "Two-Step (Bassa Temp.)"
    elif "tb" in name_lower:
        variant = "Standard (Bassa Temp.)"
    else:
        variant = "Standard (Alta Temp.)"
        
    return group, variant

def plot_nlp_metrics(all_nlp_results: dict, output_dir: str):
    metrics = ["P", "R", "F1", "IoU"]
    
    data_structured = {g: {v: None for v in VARIANTS} for g in GROUPS}
    for model_name, res in all_nlp_results.items():
        g, v = parse_model_name(model_name)
        if g in GROUPS and v in VARIANTS:
            data_structured[g][v] = res

    def plot_std_type(std_key, file_suffix):
        fig_combined, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, m in enumerate(metrics):
            ax = axes[idx]
            x = np.arange(len(GROUPS))
            width = 0.25
            offsets = [-width, 0, width]

            for i, var in enumerate(VARIANTS):
                vals = [data_structured[g][var][std_key][m] if data_structured[g][var] else 0.0 for g in GROUPS]
                ax.bar(x + offsets[i], vals, width, color=VARIANT_COLORS[var], edgecolor='black', label=var if idx == 0 else "")

            ax.set_ylabel('Score')
            ax.set_title(f'Metrica NLP ({std_key}): {m}', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(GROUPS, rotation=0)
            ax.grid(False, axis='x')
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            ax.set_ylim(0, 1.05)

        fig_combined.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        plt.suptitle(f"Metriche NLP - {std_key}", fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'nlp_metrics_combined_{file_suffix}.png'), bbox_inches='tight', dpi=300)
        plt.close()

        for m in metrics:
            fig_single, ax_single = plt.subplots(figsize=(8, 6))
            x = np.arange(len(GROUPS))
            
            for i, var in enumerate(VARIANTS):
                vals = [data_structured[g][var][std_key][m] if data_structured[g][var] else 0.0 for g in GROUPS]
                ax_single.bar(x + offsets[i], vals, width, color=VARIANT_COLORS[var], edgecolor='black', label=var)
            
            ax_single.set_ylabel('Score')
            ax_single.set_title(f'Metrica NLP ({std_key}): {m}', fontweight='bold')
            ax_single.set_xticks(x)
            ax_single.set_xticklabels(GROUPS, rotation=0)
            ax_single.grid(False, axis='x')
            ax_single.grid(True, axis='y', linestyle='--', alpha=0.7)
            ax_single.set_ylim(0, 1.05)
            
            ax_single.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'nlp_metric_{m}_{file_suffix}.png'), bbox_inches='tight', dpi=300)
            plt.close()

    plot_std_type("Non-Standardized", "non_std")
    plot_std_type("Standardized", "std")

def plot_macro_metrics(all_macro_results: dict, output_dir: str):
    fields = ["calories", "mass", "fat", "carbohydrates", "protein"]
    
    baseline = {
        "calories_MAE": 70.6, "calories_MAE_%": 26.1,
        "mass_MAE": 40.4, "mass_MAE_%": 18.8,
        "fat_MAE": 5.0, "fat_MAE_%": 34.2,
        "carbohydrates_MAE": 6.1, "carbohydrates_MAE_%": 31.9,
        "protein_MAE": 5.5, "protein_MAE_%": 29.5
    }

    data_structured = {g: {v: None for v in VARIANTS} for g in GROUPS}
    for model_name, res in all_macro_results.items():
        g, v = parse_model_name(model_name)
        if g in GROUPS and v in VARIANTS:
            data_structured[g][v] = res

    def plot_macro_type(metric_suffix, file_suffix, ylabel, title_prefix):
        fig_combined, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, f in enumerate(fields):
            ax = axes[idx]
            x = np.arange(len(GROUPS))
            width = 0.25
            offsets = [-width, 0, width]
            
            for i, var in enumerate(VARIANTS):
                vals = [data_structured[g][var][f"{f}_{metric_suffix}"] if data_structured[g][var] else 0.0 for g in GROUPS]
                ax.bar(x + offsets[i], vals, width, color=VARIANT_COLORS[var], edgecolor='black', label=var if idx == 0 else "")
            
            base_val = baseline[f"{f}_{metric_suffix}"]
            ax.axhline(y=base_val, color='r', linestyle='--', linewidth=2, label='Nutrition 5k' if idx == 0 else "")

            ax.set_ylabel(ylabel)
            ax.set_title(f'{title_prefix}: {f.capitalize()}', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(GROUPS, rotation=0)
            ax.grid(False, axis='x')
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        axes[5].axis('off')
        fig_combined.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)
        plt.suptitle(f"Confronto {title_prefix} dei Macronutrienti", fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'macro_{file_suffix}_combined.png'), bbox_inches='tight', dpi=300)
        plt.close()

        for f in fields:
            fig_single, ax_single = plt.subplots(figsize=(10, 8))
            x = np.arange(len(GROUPS))
            
            for i, var in enumerate(VARIANTS):
                vals = [data_structured[g][var][f"{f}_{metric_suffix}"] if data_structured[g][var] else 0.0 for g in GROUPS]
                ax_single.bar(x + offsets[i], vals, width, color=VARIANT_COLORS[var], edgecolor='black', label=var)
            
            base_val = baseline[f"{f}_{metric_suffix}"]
            ax_single.axhline(y=base_val, color='r', linestyle='--', linewidth=2, label='Nutrition 5k')

            ax_single.set_ylabel(ylabel)
            ax_single.set_title(f'{title_prefix}: {f.capitalize()}', fontweight='bold')
            ax_single.set_xticks(x)
            ax_single.set_xticklabels(GROUPS, rotation=0)
            ax_single.grid(False, axis='x')
            ax_single.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            ax_single.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'macro_{file_suffix}_{f}.png'), bbox_inches='tight', dpi=300)
            plt.close()

    plot_macro_type("MAE", "mae_abs", "Errore Medio Assoluto (Valore)", "MAE Assoluto")
    plot_macro_type("MAE_%", "mae_perc", "Errore Medio Assoluto (%)", "MAE Percentuale")


def print_summary_tables(all_nlp_results: dict, all_macro_results: dict):
    print("\n" + "="*120)
    print(" " * 45 + "TABELLA RIASSUNTIVA METRICHE")
    print("="*120)

    baseline = {
        "calories_MAE": 70.6, "calories_MAE_%": 26.1,
        "mass_MAE": 40.4, "mass_MAE_%": 18.8,
        "fat_MAE": 5.0, "fat_MAE_%": 34.2,
        "carbohydrates_MAE": 6.1, "carbohydrates_MAE_%": 31.9,
        "protein_MAE": 5.5, "protein_MAE_%": 29.5
    }

    models = list(all_macro_results.keys())
    models_sorted = []
    for g in GROUPS:
        for v in VARIANTS:
            for m in models:
                mg, mv = parse_model_name(m)
                if mg == g and mv == v:
                    models_sorted.append(m)
    
    if all_nlp_results:
        print("\n--- METRICHE NLP ---")
        header = f"{'Gruppo':<10} | {'Variante':<25} | {'F1':<6} | {'F1 (Std)':<8} | {'IoU':<6} | {'IoU (Std)':<9}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        for mod in models_sorted:
            if mod in all_nlp_results:
                g, v = parse_model_name(mod)
                res = all_nlp_results[mod]
                f1_ns = res["Non-Standardized"]["F1"]
                f1_s = res["Standardized"]["F1"]
                iou_ns = res["Non-Standardized"]["IoU"]
                iou_s = res["Standardized"]["IoU"]
                print(f"{g:<10} | {v:<25} | {f1_ns:.4f} | {f1_s:.4f}   | {iou_ns:.4f} | {iou_s:.4f}")
        print("-" * len(header))

    if all_macro_results:
        print("\n--- METRICHE MACRONUTRIENTI (MAE e MAE%) ---")
        header = f"{'Gruppo':<10} | {'Variante':<25} | {'Calorie':<15} | {'Massa':<15} | {'Grassi':<15} | {'Carbo':<15} | {'Proteine':<15}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        b = baseline
        cal_str = f"{b['calories_MAE']:.1f} ({b['calories_MAE_%']:.1f}%)"
        mass_str = f"{b['mass_MAE']:.1f} ({b['mass_MAE_%']:.1f}%)"
        fat_str = f"{b['fat_MAE']:.1f} ({b['fat_MAE_%']:.1f}%)"
        carb_str = f"{b['carbohydrates_MAE']:.1f} ({b['carbohydrates_MAE_%']:.1f}%)"
        prot_str = f"{b['protein_MAE']:.1f} ({b['protein_MAE_%']:.1f}%)"
        print(f"{'Baseline':<10} | {'Nutrition 5k (2D Direct)':<25} | {cal_str:<15} | {mass_str:<15} | {fat_str:<15} | {carb_str:<15} | {prot_str:<15}")
        print("-" * len(header))

        for mod in models_sorted:
            if mod in all_macro_results:
                g, v = parse_model_name(mod)
                r = all_macro_results[mod]
                cal_str = f"{r['calories_MAE']:.1f} ({r['calories_MAE_%']:.1f}%)"
                mass_str = f"{r['mass_MAE']:.1f} ({r['mass_MAE_%']:.1f}%)"
                fat_str = f"{r['fat_MAE']:.1f} ({r['fat_MAE_%']:.1f}%)"
                carb_str = f"{r['carbohydrates_MAE']:.1f} ({r['carbohydrates_MAE_%']:.1f}%)"
                prot_str = f"{r['protein_MAE']:.1f} ({r['protein_MAE_%']:.1f}%)"
                
                print(f"{g:<10} | {v:<25} | {cal_str:<15} | {mass_str:<15} | {fat_str:<15} | {carb_str:<15} | {prot_str:<15}")
        print("-" * len(header))
        print("\n")


if __name__ == "__main__":

    # ================= CONFIGURAZIONI =================
    THRESHOLD = 0.7
    OUTPUT_DIR = "data/evaluation"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Puoi mixare liberamente file derivati dal single-step e dal two-step
    PRED_JSON_PATHS = [
        "data/prediction/test_predictions_nutrition5k_gemma4_e2b_q4.json",
        "data/prediction/test_predictions_nutrition5k_gemma_e2b_q8V2.json",
        "data/prediction/test_predictions_nutrition5k_gemma4_e2b_q8_tb.json",
        "data/prediction/test_predictions_nutrition5k_twostep_gemma4_e2b_q4_tb.json",
        "data/prediction/test_predictions_nutrition5k_twostep_gemma4_e2b_q8_tb.json",
        "data/prediction/test_predictions_nutrition5k_gemma4_e4b_q4.json",
        "data/prediction/test_predictions_nutrition5k_gemma4_e4b_q8V2.json",
        "data/prediction/test_predictions_nutrition5k_gemma4_e4b_q8_tb.json",
        "data/prediction/test_predictions_nutrition5k_twostep_gemma4_e4b_q4_tb.json",
        "data/prediction/test_predictions_nutrition5k_twostep_gemma4_e4b_q8_tb.json"
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
        plot_nlp_metrics(all_nlp_results, output_dir=OUTPUT_DIR)
    if all_macro_results:
        plot_macro_metrics(all_macro_results, output_dir=OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("STAMPA TABELLA RIASSUNTIVA A TERMINALE...")
    print("=" * 70)
    print_summary_tables(all_nlp_results, all_macro_results)

    print(f"\nTutti i grafici (singoli e combinati) sono stati salvati nella cartella: {OUTPUT_DIR}")