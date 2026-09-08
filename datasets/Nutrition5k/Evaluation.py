import json
import os
import csv
import statistics
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_vocabulary(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    vocab = set()
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ing = row.get('ingr', '').strip().lower()
            if ing:
                vocab.add(ing)
    return list(vocab)


class Nutrition5kEvaluator:
    def __init__(self, pred_json_path: str, nlp_model, vocabulary: list, vocab_embeddings, threshold: float, optuna_scalars: dict = None):
        self.pred_path = pred_json_path
        self.threshold = threshold
        self.nlp_model = nlp_model
        self.vocabulary = vocabulary
        self.vocab_embeddings = vocab_embeddings
        self.optuna_scalars = optuna_scalars

        self.predictions = self._load_json(self.pred_path)
        self.similarity_log = []

    def _load_json(self, path: str) -> list:
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON file not found: {path}")
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

            # --- MODIFICATION: Pipeline auto-detection ---
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
        
        groundtruth_values_exact = {field: [] for field in DATA_FIELDNAMES}
        err_values_exact = {field: [] for field in DATA_FIELDNAMES}

        for item in self.predictions:
            gt_raw = item.get("ground_truth_ingredients", [])
            gt_list = self._clean_ingredients(gt_raw)
            gt_set = set(gt_list)

            # --- MODIFICATION: Pipeline auto-detection ---
            if "predicted_nutrition_step2" in item:
                pred_nut = item.get("predicted_nutrition_step2", {})
            else:
                pred_nut = item.get("predicted_nutrition", {})
            # -------------------------------------------------

            predicted_ingredients = pred_nut.get("visible_ingredients", []) + pred_nut.get("hidden_ingredients", [])
            pred_list = self._clean_ingredients(predicted_ingredients)
            pred_set = set(pred_list)
            
            exact_names = gt_set.intersection(pred_set)

            # --- GROUND TRUTH ---
            gt_cal = gt_mass = gt_fat = gt_carbs = gt_prot = 0.0
            gt_cal_ex = gt_mass_ex = gt_fat_ex = gt_carbs_ex = gt_prot_ex = 0.0
            
            for ing in gt_raw:
                name = str(ing.get("name", "")).strip().lower()
                mass = ing.get("weight_grams", 0.0)
                macros = ing.get("macronutrients", {})
                if not isinstance(macros, dict): macros = {}
                cal = macros.get("calories", 0.0)
                fat = macros.get("fat", 0.0)
                carbs = macros.get("carbohydrates", 0.0)
                prot = macros.get("protein", 0.0)
                
                gt_mass += mass; gt_cal += cal; gt_fat += fat; gt_carbs += carbs; gt_prot += prot
                if name in exact_names:
                    gt_mass_ex += mass; gt_cal_ex += cal; gt_fat_ex += fat; gt_carbs_ex += carbs; gt_prot_ex += prot

            # --- PREDICTIONS ---
            pred_cal = pred_mass = pred_fat = pred_carbs = pred_prot = 0.0
            pred_cal_ex = pred_mass_ex = pred_fat_ex = pred_carbs_ex = pred_prot_ex = 0.0

            for ing in predicted_ingredients:
                name = str(ing.get("name", "")).strip().lower()
                mass = ing.get("weight_grams", 0.0)
                macros = ing.get("macronutrients", {})
                if not isinstance(macros, dict): macros = {}
                cal = macros.get("calories", 0.0)
                fat = macros.get("fat", 0.0)
                carbs = macros.get("carbohydrates", 0.0)
                prot = macros.get("protein", 0.0)
                
                pred_mass += mass; pred_cal += cal; pred_fat += fat; pred_carbs += carbs; pred_prot += prot
                if name in exact_names:
                    pred_mass_ex += mass; pred_cal_ex += cal; pred_fat_ex += fat; pred_carbs_ex += carbs; pred_prot_ex += prot

            # --- COMPILING RESULTS ---
            gt_vals = {"calories": gt_cal, "mass": gt_mass, "fat": gt_fat, "carbohydrates": gt_carbs, "protein": gt_prot}
            pred_vals = {"calories": pred_cal, "mass": pred_mass, "fat": pred_fat, "carbohydrates": pred_carbs, "protein": pred_prot}
            
            gt_vals_ex = {"calories": gt_cal_ex, "mass": gt_mass_ex, "fat": gt_fat_ex, "carbohydrates": gt_carbs_ex, "protein": gt_prot_ex}
            pred_vals_ex = {"calories": pred_cal_ex, "mass": pred_mass_ex, "fat": pred_fat_ex, "carbohydrates": pred_carbs_ex, "protein": pred_prot_ex}
            
            if self.optuna_scalars:
                scale_mass = self.optuna_scalars.get("scale_mass", 1.0)
                scale_cal = self.optuna_scalars.get("scale_cal", 1.0)
                scale_fat = self.optuna_scalars.get("scale_fat", 1.0)
                scale_carbs = self.optuna_scalars.get("scale_carbs", 1.0)
                scale_protein = self.optuna_scalars.get("scale_protein", 1.0)
                
                pred_vals["mass"] *= scale_mass
                pred_vals["calories"] *= scale_cal
                pred_vals["fat"] *= scale_fat
                pred_vals["carbohydrates"] *= scale_carbs
                pred_vals["protein"] *= scale_protein
                
                pred_vals_ex["mass"] *= scale_mass
                pred_vals_ex["calories"] *= scale_cal
                pred_vals_ex["fat"] *= scale_fat
                pred_vals_ex["carbohydrates"] *= scale_carbs
                pred_vals_ex["protein"] *= scale_protein

            for field in DATA_FIELDNAMES:
                groundtruth_values[field].append(gt_vals[field])
                err_values[field].append(abs(pred_vals[field] - gt_vals[field]))
                
                groundtruth_values_exact[field].append(gt_vals_ex[field])
                err_values_exact[field].append(abs(pred_vals_ex[field] - gt_vals_ex[field]))

        output_stats = {"overall": {}, "exact_match": {}}
        for field in DATA_FIELDNAMES:
            # Overall
            mean_err = statistics.mean(err_values[field])
            mean_gt = statistics.mean(groundtruth_values[field])
            output_stats["overall"][f"{field}_MAE"] = mean_err
            output_stats["overall"][f"{field}_MAE_%"] = (100 * mean_err / mean_gt) if mean_gt > 0 else 0.0
            
            # Exact Match
            if len(err_values_exact[field]) > 0:
                mean_err_ex = statistics.mean(err_values_exact[field])
                mean_gt_ex = statistics.mean(groundtruth_values_exact[field])
            else:
                mean_err_ex = 0.0
                mean_gt_ex = 0.0
            output_stats["exact_match"][f"{field}_MAE"] = mean_err_ex
            output_stats["exact_match"][f"{field}_MAE_%"] = (100 * mean_err_ex / mean_gt_ex) if mean_gt_ex > 0 else 0.0

        return output_stats

    def export_similarity_log_to_json(self, filename: str = "similarity_analysis.json"):
        """
        Exports the similarity log to a structured JSON file grouped by recipe_id.
        """
        if not self.similarity_log:
            print("No log to export. Run evaluate_nlp_metrics() first.")
            return

        json_output = {}
        for log in self.similarity_log:
            rid = log["recipe_id"]
            if rid not in json_output:
                json_output[rid] = {}

            json_output[rid][log["predicted"]] = {
                "best_vocab_match": log["best_vocab_match"],
                "max_similarity_score": log["max_similarity_score"],
                "threshold_passed": log["threshold_passed"]
            }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4, ensure_ascii=False)

    def print_similarity_summary(self):
        if not self.similarity_log:
            return

        total = len(self.similarity_log)
        passed = sum(1 for log in self.similarity_log if log["threshold_passed"])
        avg_score = sum(log["max_similarity_score"] for log in self.similarity_log) / total

        print("\n--- NLP Similarity Analysis Summary ---")
        print(f"Total comparisons made: {total}")
        print(f"Standardized ingredients (above threshold {self.threshold}): {passed} ({(passed / total) * 100:.1f}%)")
        print(f"Unchanged ingredients: {total - passed} ({((total - passed) / total) * 100:.1f}%)")
        print(f"Average maximum similarity score (across all samples): {avg_score:.4f}")


# ==========================================
# GRAPH FUNCTIONS
# ==========================================

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

GROUPS = ['E2B - Q4', 'E2B - Q8', 'E4B - Q4', 'E4B - Q8']


def get_dynamic_variants_and_colors():
    import glob
    files = glob.glob("data/prediction/one_example/*.json")
    # Includiamo YOLO ma rimuoviamo Optuna
    requested = [f for f in files if "optuna" not in f.lower()]
    var_set = set()
    for f in requested:
        import os
        filename = os.path.basename(f)
        _, v = parse_model_name(filename.replace('test_predictions_nutrition5k_', '').replace('.json', ''))
        if v != "Unknown" and "Optuna" not in v:
            var_set.add(v)

    variants = sorted(list(var_set))
    cmap = plt.get_cmap('tab10')
    colors = {v: mcolors.to_hex(cmap(i)) for i, v in enumerate(variants)}
    return variants, colors

VARIANTS, VARIANT_COLORS = [], {}


def parse_model_name(name: str):
    # (Mantieni la tua funzione parse_model_name identica a come l'hai già)
    name_lower = name.lower()
    group = "Unknown"
    if "e2b" in name_lower and "q4" in name_lower: group = "E2B - Q4"
    elif "e2b" in name_lower and "q8" in name_lower: group = "E2B - Q8"
    elif "e4b" in name_lower and "q4" in name_lower: group = "E4B - Q4"
    elif "e4b" in name_lower and "q8" in name_lower: group = "E4B - Q8"

    variant = "Unknown"
    if "optuna" in name_lower and "yolo" in name_lower: variant = "YOLO Optuna"
    elif "v2hybrid" in name_lower: variant = "Hybrid v2"
    elif "hybrid" in name_lower: variant = "Hybrid v1"
    elif "twostep" in name_lower: variant = "Two-Step"
    elif "yolo" in name_lower: variant = "YOLO"
    else: variant = "Standard"

    return group, variant


def plot_nlp_metrics(all_nlp_results: dict, output_dir: str):
    metrics = ["P", "R", "F1", "IoU"]

    data_structured = {g: {v: None for v in VARIANTS} for g in GROUPS}
    for model_name, res in all_nlp_results.items():
        g, v = parse_model_name(model_name)
        if g in GROUPS and v in VARIANTS:
            data_structured[g][v] = res

    def plot_std_type(std_key, file_suffix):
        combined_metrics = ["F1", "IoU"]
        fig_combined, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes = axes.flatten()

        for idx, m in enumerate(combined_metrics):
            ax = axes[idx]
            x = np.arange(len(GROUPS))
            width = 0.2
            offsets = [(i - len(VARIANTS)/2 + 0.5) * width for i in range(len(VARIANTS))]

            for i, var in enumerate(VARIANTS):
                vals = [data_structured[g][var][std_key][m] if data_structured[g][var] else 0.0 for g in GROUPS]
                ax.bar(x + offsets[i], vals, width, color=VARIANT_COLORS[var], edgecolor='black',
                       label=var if idx == 0 else "")

            ax.set_ylabel('Score')
            ax.set_title(f'NLP Metrics ({std_key}): {m}', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(GROUPS, rotation=0)
            ax.grid(False, axis='x')
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            ax.set_ylim(0, 1.05)

        fig_combined.legend(loc='lower center', bbox_to_anchor=(0.5, -0.10), ncol=3)
        plt.suptitle(f"NLP Metrics - {std_key}", fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'nlp_metrics_combined_{file_suffix}.png'), bbox_inches='tight', dpi=300)
        plt.close()

        for m in metrics:
            fig_single, ax_single = plt.subplots(figsize=(8, 6))
            x = np.arange(len(GROUPS))
            width = 0.2
            offsets = [(i - len(VARIANTS)/2 + 0.5) * width for i in range(len(VARIANTS))]
            for i, var in enumerate(VARIANTS):
                vals = [data_structured[g][var][std_key][m] if data_structured[g][var] else 0.0 for g in GROUPS]
                ax_single.bar(x + offsets[i], vals, width, color=VARIANT_COLORS[var], edgecolor='black', label=var)

            ax_single.set_ylabel('Score')
            ax_single.set_title(f'NLP Metric ({std_key}): {m}', fontweight='bold')
            ax_single.set_xticks(x)
            ax_single.set_xticklabels(GROUPS, rotation=0)
            ax_single.grid(False, axis='x')
            ax_single.grid(True, axis='y', linestyle='--', alpha=0.7)
            ax_single.set_ylim(0, 1.05)

            ax_single.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'nlp_metric_{m}_{file_suffix}.png'), bbox_inches='tight', dpi=300)
            plt.close()

    plot_std_type("Non-Standardized", "non_std")
    plot_std_type("Standardized", "std")


def plot_macro_metrics(all_macro_results: dict, output_dir: str, plot_baseline: bool = True):
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

    def plot_macro_type(metric_suffix, file_suffix, ylabel, title_prefix, log_scale=False):
        fig_combined, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, f in enumerate(fields):
            ax = axes[idx]
            x = np.arange(len(GROUPS))
            width = 0.15
            offsets = [(i - len(VARIANTS)/2 + 0.5) * width for i in range(len(VARIANTS))]

            for i, var in enumerate(VARIANTS):
                vals = [data_structured[g][var][f"{f}_{metric_suffix}"] if data_structured[g][var] else 0.0 for g in
                        GROUPS]
                vals = [max(v, 0.01) if log_scale else v for v in vals]
                ax.bar(x + offsets[i], vals, width, color=VARIANT_COLORS[var], edgecolor='black',
                       label=var if idx == 0 else "")

            if plot_baseline:
                base_val = baseline[f"{f}_{metric_suffix}"]
                ax.axhline(y=base_val, color='r', linestyle='--', linewidth=2,
                           label='Nutrition 5k (2D direct)' if idx == 0 else "")

            ax.set_ylabel(ylabel + (" (Log Scale)" if log_scale else ""))
            if log_scale:
                ax.set_yscale('log')
            ax.set_title(f'{title_prefix}: {f.capitalize()}', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(GROUPS, rotation=0)
            ax.grid(False, axis='x')
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        axes[5].axis('off')
        fig_combined.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=5)
        title_suffix = " (Log Scale)" if log_scale else ""
        plt.suptitle(f"{title_prefix} Comparison of Macronutrients{title_suffix}", fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        file_suffix_out = f"{file_suffix}_log" if log_scale else file_suffix
        plt.savefig(os.path.join(output_dir, f'macro_{file_suffix_out}_combined.png'), bbox_inches='tight', dpi=300)
        plt.close()

        for f in fields:
            fig_single, ax_single = plt.subplots(figsize=(10, 8))
            x = np.arange(len(GROUPS))
            width = 0.15
            offsets = [(i - len(VARIANTS)/2 + 0.5) * width for i in range(len(VARIANTS))]
            for i, var in enumerate(VARIANTS):
                vals = [data_structured[g][var][f"{f}_{metric_suffix}"] if data_structured[g][var] else 0.0 for g in
                        GROUPS]
                vals = [max(v, 0.01) if log_scale else v for v in vals]
                ax_single.bar(x + offsets[i], vals, width, color=VARIANT_COLORS[var], edgecolor='black', label=var)

            if plot_baseline:
                base_val = baseline[f"{f}_{metric_suffix}"]
                ax_single.axhline(y=base_val, color='r', linestyle='--', linewidth=2, label='Nutrition 5k (2D direct)')

            ax_single.set_ylabel(ylabel + (" (Log Scale)" if log_scale else ""))
            if log_scale:
                ax_single.set_yscale('log')
            ax_single.set_title(f'{title_prefix}: {f.capitalize()}', fontweight='bold')
            ax_single.set_xticks(x)
            ax_single.set_xticklabels(GROUPS, rotation=0)
            ax_single.grid(False, axis='x')
            ax_single.grid(True, axis='y', linestyle='--', alpha=0.7)

            ax_single.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4)
            plt.tight_layout()
            file_suffix_out = f"{file_suffix}_log" if log_scale else file_suffix
            plt.savefig(os.path.join(output_dir, f'macro_{file_suffix_out}_{f}.png'), bbox_inches='tight', dpi=300)
            plt.close()

    plot_macro_type("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE", log_scale=False)
    plot_macro_type("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE", log_scale=False)
    plot_macro_type("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE", log_scale=True)
    plot_macro_type("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE", log_scale=True)


def plot_overlay_macro_metrics(all_macro_results: dict, output_dir: str):
    fields = ["calories", "mass", "fat", "carbohydrates", "protein"]
    metric_types = {"MAE": "Mean Absolute Error", "MAE_%": "Mean Absolute Error (%)"}

    data_structured = {g: {v: None for v in VARIANTS} for g in GROUPS}
    for model_name, res in all_macro_results.items():
        g, v = parse_model_name(model_name)
        if g in GROUPS and v in VARIANTS:
            data_structured[g][v] = res

    os.makedirs(output_dir, exist_ok=True)

    def plot_overlay_type(metric_suffix, file_suffix, ylabel, log_scale=False):
        fig_combined, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, f in enumerate(fields):
            ax = axes[idx]
            x = np.arange(len(GROUPS))
            width = 0.08
            offsets = [(i - (len(VARIANTS)*2)/2 + 0.5) * width for i in range(len(VARIANTS)*2)]

            for i, var in enumerate(VARIANTS):
                overall_vals = [data_structured[g][var]["overall"][f"{f}_{metric_suffix}"] if data_structured[g][var] else 0.0 for g in GROUPS]
                exact_vals = [data_structured[g][var]["exact_match"][f"{f}_{metric_suffix}"] if data_structured[g][var] else 0.0 for g in GROUPS]
                
                overall_vals = [max(v, 0.01) if log_scale else v for v in overall_vals]
                exact_vals = [max(v, 0.01) if log_scale else v for v in exact_vals]

                ax.bar(x + offsets[i*2], overall_vals, width, color=VARIANT_COLORS[var], edgecolor='black', 
                       label=f"{var} - Overall" if idx == 0 else "")
                ax.bar(x + offsets[i*2 + 1], exact_vals, width, color=VARIANT_COLORS[var], edgecolor='black', hatch='//', alpha=0.8, 
                       label=f"{var} - Exact Match" if idx == 0 else "")

            ax.set_ylabel(ylabel + (" (Log Scale)" if log_scale else ""))
            if log_scale:
                ax.set_yscale('log')
            ax.set_title(f'Overlay: {f.capitalize()}', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(GROUPS, rotation=0)
            ax.grid(False, axis='x')
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        axes[5].axis('off')
        fig_combined.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=5)
        file_suffix_base = "mae_perc" if metric_suffix == "MAE_%" else "mae_abs"
        file_suffix_out = f"{file_suffix_base}_log" if log_scale else file_suffix_base
        title_suffix = " (Log Scale)" if log_scale else ""
        plt.suptitle(f"Overall vs Exact Match: {ylabel}{title_suffix}", fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'overlay_{file_suffix_out}_combined.png'), bbox_inches='tight', dpi=300)
        plt.close()

        # Generate individual plots as well
        for f in fields:
            fig_single, ax_single = plt.subplots(figsize=(10, 8))
            x = np.arange(len(GROUPS))
            width = 0.08
            offsets = [(i - (len(VARIANTS)*2)/2 + 0.5) * width for i in range(len(VARIANTS)*2)]
            for i, var in enumerate(VARIANTS):
                overall_vals = [data_structured[g][var]["overall"][f"{f}_{metric_suffix}"] if data_structured[g][var] else 0.0 for g in GROUPS]
                exact_vals = [data_structured[g][var]["exact_match"][f"{f}_{metric_suffix}"] if data_structured[g][var] else 0.0 for g in GROUPS]
                
                overall_vals = [max(v, 0.01) if log_scale else v for v in overall_vals]
                exact_vals = [max(v, 0.01) if log_scale else v for v in exact_vals]

                ax_single.bar(x + offsets[i*2], overall_vals, width, color=VARIANT_COLORS[var], edgecolor='black', label=f"{var} - Overall")
                ax_single.bar(x + offsets[i*2 + 1], exact_vals, width, color=VARIANT_COLORS[var], edgecolor='black', hatch='//', alpha=0.8, label=f"{var} - Exact Match")

            ax_single.set_ylabel(ylabel + (" (Log Scale)" if log_scale else ""))
            if log_scale:
                ax_single.set_yscale('log')
            ax_single.set_title(f'Overall vs Exact Match: {f.capitalize()}', fontweight='bold')
            ax_single.set_xticks(x)
            ax_single.set_xticklabels(GROUPS, rotation=0)
            ax_single.grid(False, axis='x')
            ax_single.grid(True, axis='y', linestyle='--', alpha=0.7)

            ax_single.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'overlay_{file_suffix_out}_{f}.png'), bbox_inches='tight', dpi=300)
            plt.close()

    plot_overlay_type("MAE", "mae_abs", "Mean Absolute Error (Value)", log_scale=False)
    plot_overlay_type("MAE_%", "mae_perc", "Mean Absolute Error (%)", log_scale=False)
    plot_overlay_type("MAE", "mae_abs", "Mean Absolute Error (Value)", log_scale=True)
    plot_overlay_type("MAE_%", "mae_perc", "Mean Absolute Error (%)", log_scale=True)


# --- NUOVE FUNZIONI DI SUPPORTO PER IL GRAFICO THRESHOLD ---
def get_marker(group_name: str) -> str:
    if 'E2B' in group_name:
        return 'o'  # Cerchio per modelli E2B
    elif 'E4B' in group_name:
        return '^'  # Triangolo per modelli E4B
    return 'x'  # Fallback


def get_linestyle(group_name: str) -> str:
    if 'Q4' in group_name:
        return '-'  # Linea continua per la quantizzazione Q4
    elif 'Q8' in group_name:
        return '--'  # Tratteggiato per la quantizzazione Q8
    return '-'


# -----------------------------------------------------------

def plot_threshold_study(thresholds, study_results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for model_name, metrics in study_results.items():
        g, v = parse_model_name(model_name)

        # color mapping
        color = VARIANT_COLORS.get(v, '#333333')

        # marker and style mapping
        marker = get_marker(g)
        linestyle = get_linestyle(g)

        axes[0].plot(thresholds, metrics['F1'], color=color,
                     linestyle=linestyle, marker=marker, markersize=6, linewidth=1.5)
        axes[1].plot(thresholds, metrics['IoU'], color=color,
                     linestyle=linestyle, marker=marker, markersize=6, linewidth=1.5)

    axes[0].set_title('Standardized F1 vs Threshold', fontweight='bold')
    axes[0].set_xlabel('Similarity Threshold')
    axes[0].set_ylabel('F1 Score')
    axes[0].invert_xaxis()
    axes[0].grid(True, linestyle='--', alpha=0.7)

    axes[1].set_title('Standardized IoU vs Threshold', fontweight='bold')
    axes[1].set_xlabel('Similarity Threshold')
    axes[1].set_ylabel('IoU Score')
    axes[1].invert_xaxis()
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # legend
    import matplotlib.lines as mlines
    legend_elements = [
        # Simboli Modelli (Colonna 1)
        mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=8, label='E2B Model'),
        mlines.Line2D([], [], color='gray', marker='^', linestyle='None', markersize=8, label='E4B Model'),
        # Quantizzazioni (Colonna 2)
        mlines.Line2D([], [], color='gray', linestyle='-', linewidth=2, label='Q4 Quantization'),
        mlines.Line2D([], [], color='gray', linestyle='--', linewidth=2, label='Q8 Quantization'),
        # Tecniche di Prompting (Colonna 3)
        # Dynamic variants
] + [mlines.Line2D([], [], color=VARIANT_COLORS[var], linewidth=4, label=var) for var in VARIANTS]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=4)

    plt.suptitle("Influence of Similarity Threshold on NLP Metrics", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nlp_threshold_study.png'), bbox_inches='tight', dpi=300)
    plt.close()


def print_summary_tables(all_nlp_results: dict, all_macro_results: dict):
    print("\n" + "=" * 120)
    print(" " * 45 + "METRICS SUMMARY TABLE")
    print("=" * 120)

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
        print("\n--- NLP METRICS ---")
        header = f"{'Group':<10} | {'Variant':<25} | {'F1':<6} | {'F1 (Std)':<8} | {'IoU':<6} | {'IoU (Std)':<9}"
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
        for subset in ["overall", "exact_match"]:
            title = "MACRONUTRIENTS METRICS (OVERALL)" if subset == "overall" else "MACRONUTRIENTS METRICS (EXACT MATCH)"
            print(f"\n--- {title} ---")
            header = f"{'Group':<10} | {'Variant':<25} | {'Calories':<15} | {'Mass':<15} | {'Fat':<15} | {'Carbs':<15} | {'Protein':<15}"
            print("-" * len(header))
            print(header)
            print("-" * len(header))

            if subset == "overall":
                b = baseline
                cal_str = f"{b['calories_MAE']:.1f} ({b['calories_MAE_%']:.1f}%)"
                mass_str = f"{b['mass_MAE']:.1f} ({b['mass_MAE_%']:.1f}%)"
                fat_str = f"{b['fat_MAE']:.1f} ({b['fat_MAE_%']:.1f}%)"
                carb_str = f"{b['carbohydrates_MAE']:.1f} ({b['carbohydrates_MAE_%']:.1f}%)"
                prot_str = f"{b['protein_MAE']:.1f} ({b['protein_MAE_%']:.1f}%)"
                print(
                    f"{'Baseline':<10} | {'Nutrition 5k (2D Direct)':<25} | {cal_str:<15} | {mass_str:<15} | {fat_str:<15} | {carb_str:<15} | {prot_str:<15}")
                print("-" * len(header))

            for mod in models_sorted:
                if mod in all_macro_results:
                    g, v = parse_model_name(mod)
                    r = all_macro_results[mod][subset]
                    cal_str = f"{r['calories_MAE']:.1f} ({r['calories_MAE_%']:.1f}%)"
                    mass_str = f"{r['mass_MAE']:.1f} ({r['mass_MAE_%']:.1f}%)"
                    fat_str = f"{r['fat_MAE']:.1f} ({r['fat_MAE_%']:.1f}%)"
                    carb_str = f"{r['carbohydrates_MAE']:.1f} ({r['carbohydrates_MAE_%']:.1f}%)"
                    prot_str = f"{r['protein_MAE']:.1f} ({r['protein_MAE_%']:.1f}%)"

                    print(
                        f"{g:<10} | {v:<25} | {cal_str:<15} | {mass_str:<15} | {fat_str:<15} | {carb_str:<15} | {prot_str:<15}")
            print("-" * len(header))
            print("\n")


FEW_SHOT_VARIANTS, FEW_SHOT_COLORS = [], {}


def init_dynamic_globals():
    global VARIANTS, VARIANT_COLORS, FEW_SHOT_VARIANTS, FEW_SHOT_COLORS
    VARIANTS, VARIANT_COLORS = get_dynamic_variants_and_colors()

    # Mappa dei colori Gerarchica
    color_map = {
        'Standard': ['#99c2ff', '#3385ff', '#0047b3'],  # Blu
        'Two-Step': ['#ffb3b3', '#ff4d4d', '#b30000'],  # Rosso
        'Hybrid v1': ['#a3e0a3', '#47d147', '#1f7a1f'],  # Verde
        'Hybrid v2': ['#d1b3ff', '#9933ff', '#4d0099'],  # Viola
        'YOLO': ['#ffcc99', '#ff9933', '#cc6600']        # Arancione
    }

    fs_vars = []
    FEW_SHOT_COLORS = {}

    for p_idx, p in enumerate(['1 Example', '5 Examples', '10 Examples']):
        for v in VARIANTS:
            if 'Standard' in v:
                name = f"{p} (Standard)"
                fs_vars.append(name)
                FEW_SHOT_COLORS[name] = color_map['Standard'][p_idx]
            elif 'Two-Step' in v:
                name = f"{p} (Two-Step)"
                fs_vars.append(name)
                FEW_SHOT_COLORS[name] = color_map['Two-Step'][p_idx]
            elif 'Hybrid v2' in v:
                name = f"{p} (Hybrid v2)"
                fs_vars.append(name)
                FEW_SHOT_COLORS[name] = color_map['Hybrid v2'][p_idx]
            elif 'Hybrid v1' in v:
                name = f"{p} (Hybrid v1)"
                fs_vars.append(name)
                FEW_SHOT_COLORS[name] = color_map['Hybrid v1'][p_idx]
            elif 'YOLO' in v and 'Optuna' not in v:
                name = f"{p} (YOLO)"
                fs_vars.append(name)
                FEW_SHOT_COLORS[name] = color_map['YOLO'][p_idx]

    FEW_SHOT_VARIANTS = list(dict.fromkeys(fs_vars))

def plot_few_shot_metrics(all_results: dict, output_dir: str):
    nlp_dir = os.path.join(output_dir, "nlp_metrics")
    mae_overall_dir = os.path.join(output_dir, "mae_metrics", "overall")
    mae_exact_dir = os.path.join(output_dir, "mae_metrics", "exact_match")

    os.makedirs(nlp_dir, exist_ok=True)
    os.makedirs(mae_overall_dir, exist_ok=True)
    os.makedirs(mae_exact_dir, exist_ok=True)
    
    # 1. NLP Metrics
    # 1. NLP Metrics
    def plot_few_shot_nlp_type(std_key, file_suffix):
        import matplotlib.lines as mlines
        metrics_nlp = ["P", "R", "F1", "IoU"]

        # Mappature per il Dot Plot
        # Mappature per il Dot Plot
        strategies = ['Standard', 'Two-Step', 'Hybrid v1', 'Hybrid v2', 'YOLO']
        examples = ['1 Example', '5 Examples', '10 Examples']

        strat_colors = {
            'Standard': '#3385ff',
            'Two-Step': '#ff4d4d',
            'Hybrid v1': '#47d147',
            'Hybrid v2': '#9933ff',
            'YOLO': '#ff9933'  # Arancione
        }
        strat_markers = {
            'Standard': 'o',
            'Two-Step': 's',
            'Hybrid v1': '^',
            'Hybrid v2': 'D',
            'YOLO': 'P'  # Simbolo della croce riempita (Plus)
        }
        # Offset sull'asse X per raggruppare visivamente gli esempi
        ex_offsets = {'1 Example': -0.22, '5 Examples': 0.0, '10 Examples': 0.22}

        def draw_nlp_dot_plot(ax, m):
            for grp_idx, g in enumerate(GROUPS):
                # Linee divisorie verticali
                if grp_idx > 0:
                    ax.axvline(x=grp_idx - 0.5, color='#e0e0e0', linestyle='-', linewidth=1, zorder=0)

                for ex_name in examples:
                    x_pos = grp_idx + ex_offsets[ex_name]

                    for strat in strategies:
                        var_name = f"{ex_name} ({strat})"
                        res = all_results.get(g, {}).get(var_name)

                        val = 0.0
                        if res and "nlp" in res and std_key in res["nlp"]:
                            val = res["nlp"][std_key][m]

                        if val > 0.0:
                            # Disegno del marker con alpha per gestire overplotting
                            ax.scatter(x_pos, val, color=strat_colors[strat], marker=strat_markers[strat],
                                       s=90, edgecolors='white', linewidth=0.8, alpha=0.85, zorder=4)

            # Tipografia e assi
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title(f'NLP {std_key}: {m}', fontsize=14, fontweight='bold')
            ax.set_xticks(np.arange(len(GROUPS)))
            ax.set_xticklabels(GROUPS, fontsize=12, fontweight='bold')
            ax.tick_params(axis='y', labelsize=11)
            ax.grid(False, axis='x')
            ax.grid(True, axis='y', linestyle=':', color='#cccccc', zorder=0)
            ax.set_xlim(-0.5, len(GROUPS) - 0.5)
            # Fissiamo l'asse Y tra 0 e 1 (con un piccolo margine in alto) per le metriche NLP
            ax.set_ylim(0, 1.05)

        # -----------------------------------------------------------------
        # COSTRUZIONE GRAFICO COMBINATO (F1 e IoU)
        # -----------------------------------------------------------------
        combined_metrics = ["F1", "IoU"]
        fig_nlp, axes_nlp = plt.subplots(1, 2, figsize=(14, 6))
        for idx, m in enumerate(combined_metrics):
            draw_nlp_dot_plot(axes_nlp[idx], m)

        # Creazione di una legenda unificata orizzontale
        legend_elements = []
        legend_elements.append(mlines.Line2D([], [], color='none', label='STRATEGIA:'))
        for strat in strategies:
            legend_elements.append(mlines.Line2D([], [], color='none', marker=strat_markers[strat],
                                                 markerfacecolor=strat_colors[strat], markeredgecolor='white',
                                                 markersize=9, label=strat))

        legend_elements.append(mlines.Line2D([], [], color='none', label='  |  ESEMPI (X):'))
        legend_elements.append(mlines.Line2D([], [], color='none', marker='|', markeredgecolor='black', markersize=10,
                                             label='Sinistra = 1'))
        legend_elements.append(
            mlines.Line2D([], [], color='none', marker='|', markeredgecolor='black', markersize=10, label='Centro = 5'))
        legend_elements.append(mlines.Line2D([], [], color='none', marker='|', markeredgecolor='black', markersize=10,
                                             label='Destra = 10'))

        fig_nlp.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
                       ncol=9, fontsize=10, frameon=False, columnspacing=0.8)

        plt.suptitle(f"Few-Shot Comparison - NLP Metrics ({std_key})", fontsize=16, fontweight='bold', y=0.98)
        # Margine inferiore aumentato per fare spazio alla legenda
        plt.tight_layout(rect=[0, 0.10, 1, 0.95])
        plt.savefig(os.path.join(nlp_dir, f'few_shot_nlp_combined_{file_suffix}.png'), bbox_inches='tight', dpi=300)
        plt.close()

        # -----------------------------------------------------------------
        # COSTRUZIONE GRAFICI SINGOLI
        # -----------------------------------------------------------------
        for m in metrics_nlp:
            fig_single, ax_single = plt.subplots(figsize=(8, 6))
            draw_nlp_dot_plot(ax_single, m)

            # Legenda compatta strutturata su 2 righe per i grafici stretti
            fig_single.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                              ncol=5, fontsize=10, frameon=False)

            plt.tight_layout(rect=[0, 0.15, 1, 0.95])
            plt.savefig(os.path.join(nlp_dir, f'few_shot_nlp_{m}_{file_suffix}.png'), bbox_inches='tight', dpi=300)
            plt.close()

    plot_few_shot_nlp_type("Standardized", "std")
    plot_few_shot_nlp_type("Non-Standardized", "non_std")

    # 2. Macro metrics
    fields = ["calories", "mass", "fat", "carbohydrates", "protein"]
    
    baseline = {
        "calories_MAE": 70.6, "calories_MAE_%": 26.1,
        "mass_MAE": 40.4, "mass_MAE_%": 18.8,
        "fat_MAE": 5.0, "fat_MAE_%": 34.2,
        "carbohydrates_MAE": 6.1, "carbohydrates_MAE_%": 31.9,
        "protein_MAE": 5.5, "protein_MAE_%": 29.5
    }

    def plot_few_shot_macro_type(metric_suffix, file_suffix, ylabel, title_prefix, subset="overall", plot_baseline=True,
                                 log_scale=False):
        import matplotlib.lines as mlines
        from matplotlib.ticker import ScalarFormatter

        out_dir = mae_overall_dir if subset == "overall" else mae_exact_dir
        fig_macro, axes_macro = plt.subplots(2, 3, figsize=(18, 12))
        axes_macro = axes_macro.flatten()

        # Mappature per il Dot Plot
        # Mappature per il Dot Plot
        strategies = ['Standard', 'Two-Step', 'Hybrid v1', 'Hybrid v2', 'YOLO']
        examples = ['1 Example', '5 Examples', '10 Examples']

        strat_colors = {
            'Standard': '#3385ff',
            'Two-Step': '#ff4d4d',
            'Hybrid v1': '#47d147',
            'Hybrid v2': '#9933ff',
            'YOLO': '#ff9933'  # Arancione
        }
        strat_markers = {
            'Standard': 'o',
            'Two-Step': 's',
            'Hybrid v1': '^',
            'Hybrid v2': 'D',
            'YOLO': 'P'  # Simbolo della croce riempita (Plus)
        }
        # Offset sull'asse X per raggruppare visivamente gli esempi
        ex_offsets = {'1 Example': -0.22, '5 Examples': 0.0, '10 Examples': 0.22}

        # Funzione helper per disegnare il Dot Plot sia sui pannelli combinati che singoli
        def draw_dot_plot(ax, f):
            # Linea di Baseline e Testo inline
            if plot_baseline:
                base_val = baseline[f"{f}_{metric_suffix}"]
                ax.axhline(y=base_val, color='gray', linestyle='--', linewidth=1.2, zorder=1)

                # Posiziona l'etichetta del baseline a sinistra
                text_y = base_val * 1.05 if log_scale else base_val + (base_val * 0.02)
                ax.text(x=-0.45, y=text_y, s=' Nutrition 5k', color='gray', fontsize=11,
                        va='bottom', fontweight='bold', zorder=2)

            for grp_idx, g in enumerate(GROUPS):
                # Linee divisorie verticali chiare tra i gruppi di modelli
                if grp_idx > 0:
                    ax.axvline(x=grp_idx - 0.5, color='lightgray', linestyle=':', linewidth=1.5, zorder=0)

                for ex_name in examples:
                    x_pos = grp_idx + ex_offsets[ex_name]

                    for strat in strategies:
                        var_name = f"{ex_name} ({strat})"
                        res = all_results.get(g, {}).get(var_name)

                        val = 0.0
                        if res and "macro" in res and subset in res["macro"]:
                            val = res["macro"][subset][f"{f}_{metric_suffix}"]

                        val = max(val, 0.01) if log_scale else val

                        if val > 0.01 or not log_scale:
                            # Evidenzia chi batte il baseline
                            is_below = plot_baseline and (val < base_val)
                            edgecolor = 'black' if is_below else 'white'
                            lw = 1.5 if is_below else 0.5
                            size = 110 if is_below else 70

                            ax.scatter(x_pos, val, color=strat_colors[strat], marker=strat_markers[strat],
                                       s=size, edgecolors=edgecolor, linewidth=lw, zorder=4)

            # Tipografia migliorata (Punti 4, 6 e 7)
            ax.set_ylabel(ylabel, fontsize=12)
            if log_scale:
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(ScalarFormatter())

            ax.set_title(f'{f.capitalize()}', fontsize=14, fontweight='bold')
            ax.set_xticks(np.arange(len(GROUPS)))
            ax.set_xticklabels(GROUPS, fontsize=12, fontweight='bold')
            ax.tick_params(axis='y', labelsize=11)
            ax.grid(False, axis='x')
            ax.grid(True, axis='y', linestyle='-', color='#eeeeee', zorder=0)
            ax.set_xlim(-0.5, len(GROUPS) - 0.5)

        # -----------------------------------------------------------------
        # COSTRUZIONE GRAFICO COMBINATO (2x3)
        # -----------------------------------------------------------------
        for idx, f in enumerate(fields):
            draw_dot_plot(axes_macro[idx], f)

        # 6° Riquadro: Legenda Pulita ed Esplicativa
        axes_macro[5].axis('off')
        legend_elements = []

        # Sezione: Strategie
        legend_elements.append(mlines.Line2D([], [], color='none', label='STRATEGIA (Forma e Colore):'))
        for strat in strategies:
            legend_elements.append(mlines.Line2D([], [], color='none', marker=strat_markers[strat],
                                                 markerfacecolor=strat_colors[strat], markeredgecolor='white',
                                                 markersize=9, label=f"  {strat}"))

        # Sezione: Posizionamento (Esempi)
        legend_elements.append(mlines.Line2D([], [], color='none', label=''))  # Spazio
        legend_elements.append(mlines.Line2D([], [], color='none', label='N. ESEMPI (Posizione su asse X):'))
        legend_elements.append(mlines.Line2D([], [], color='none', marker='|', markeredgecolor='black', markersize=10,
                                             label='  Sinistra = 1 Esempio'))
        legend_elements.append(mlines.Line2D([], [], color='none', marker='|', markeredgecolor='black', markersize=10,
                                             label='  Centro = 5 Esempi'))
        legend_elements.append(mlines.Line2D([], [], color='none', marker='|', markeredgecolor='black', markersize=10,
                                             label='  Destra = 10 Esempi'))

        if plot_baseline:
            legend_elements.append(mlines.Line2D([], [], color='none', label=''))  # Spazio
            legend_elements.append(mlines.Line2D([], [], color='none', marker='o', markerfacecolor='#e0e0e0',
                                                 markeredgecolor='black', linewidth=1.5, markersize=12,
                                                 label='Bordo nero = Migliore del Baseline'))

        axes_macro[5].legend(handles=legend_elements, loc='center left', fontsize=11, frameon=False, borderpad=1)

        title_suffix = " (Log Scale)" if log_scale else ""
        plt.suptitle(f"{title_prefix}{title_suffix}", fontsize=18, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        file_suffix_out = f"{file_suffix}_log" if log_scale else file_suffix
        plt.savefig(os.path.join(out_dir, f'few_shot_macro_{subset}_{file_suffix_out}_combined.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

        # -----------------------------------------------------------------
        # COSTRUZIONE GRAFICI SINGOLI
        # -----------------------------------------------------------------
        for f in fields:
            fig_single, ax_single = plt.subplots(figsize=(10, 8))
            draw_dot_plot(ax_single, f)

            # Per i singoli riquadri creiamo una legenda orizzontale compatta
            single_leg_elements = []
            for strat in strategies:
                single_leg_elements.append(mlines.Line2D([], [], color='none', marker=strat_markers[strat],
                                                         markerfacecolor=strat_colors[strat], markeredgecolor='white',
                                                         markersize=9, label=strat))

            ax_single.legend(handles=single_leg_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                             ncol=4, fontsize=11, title="Strategie (Sinistra=1 ex, Centro=5 ex, Destra=10 ex)",
                             title_fontsize=12, frameon=False)

            plt.tight_layout(rect=[0, 0.15, 1, 1])
            plt.savefig(os.path.join(out_dir, f'few_shot_macro_{subset}_{file_suffix_out}_{f}.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()

    plot_few_shot_macro_type("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Overall)", "overall", plot_baseline=True)
    plot_few_shot_macro_type("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Overall)", "overall", plot_baseline=True)
    plot_few_shot_macro_type("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Exact Match)", "exact_match", plot_baseline=False)
    plot_few_shot_macro_type("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Exact Match)", "exact_match", plot_baseline=False)

    plot_few_shot_macro_type("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Overall)", "overall", plot_baseline=True, log_scale=True)
    plot_few_shot_macro_type("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Overall)", "overall", plot_baseline=True, log_scale=True)
    plot_few_shot_macro_type("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Exact Match)", "exact_match", plot_baseline=False, log_scale=True)
    plot_few_shot_macro_type("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Exact Match)", "exact_match", plot_baseline=False, log_scale=True)

def evaluate_few_shot_learning(nlp_model, vocabulary, vocab_embeddings, threshold, base_dir="data/prediction"):
    print("\n" + "=" * 70)
    print("STARTING FEW-SHOT EVALUATION (1, 5, 10 Examples)")
    print("=" * 70)
    
    folders = ["one_example", "five_example", "ten_example"]
    folder_to_prefix = {"one_example": "1 Example", "five_example": "5 Examples", "ten_example": "10 Examples"}
    
    all_few_shot_results = {g: {v: None for v in FEW_SHOT_VARIANTS} for g in GROUPS}
    
    fs_csv_path = os.path.join("data/evaluation", "few_shot_evaluation.csv")
    fs_yolo_csv_path = os.path.join("data/evaluation", "few_shot_yolo_evaluation.csv")
    
    def load_fs_csv(path):
        if not os.path.exists(path):
            return False
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                g = row['Group']
                v = row['Variant']
                if g in all_few_shot_results and v in all_few_shot_results[g]:
                    all_few_shot_results[g][v] = {
                        "nlp": {
                            "Non-Standardized": {k: float(row[f"nlp_Non-Standardized_{k}"]) for k in ["P", "R", "F1", "IoU"]},
                            "Standardized": {k: float(row[f"nlp_Standardized_{k}"]) for k in ["P", "R", "F1", "IoU"]}
                        },
                        "macro": {
                            "overall": {f"{f_name}_{metric}": float(row[f"macro_overall_{f_name}_{metric}"]) 
                                        for f_name in ["calories", "mass", "fat", "carbohydrates", "protein"]
                                        for metric in ["MAE", "MAE_%"]},
                            "exact_match": {f"{f_name}_{metric}": float(row[f"macro_exact_match_{f_name}_{metric}"]) 
                                        for f_name in ["calories", "mass", "fat", "carbohydrates", "protein"]
                                        for metric in ["MAE", "MAE_%"]}
                        }
                    }
        return True

    loaded_fs = load_fs_csv(fs_csv_path)
    loaded_fs_yolo = load_fs_csv(fs_yolo_csv_path)
    updated_fs = False
    updated_fs_yolo = False

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.endswith('.json'):
                continue
            # RIMOSSA LA RIGA: if "yolo" in filename.lower(): continue

            pred_path = os.path.join(folder_path, filename)
            g, v_type = parse_model_name(filename)
            if g not in GROUPS:
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
            var_name_optuna = f"{prefix} (YOLO Optuna)" if is_yolo else None  # <-- RIGA DA AGGIUNGERE

            needs_base = True
            needs_optuna = False

            if is_yolo:
                if loaded_fs_yolo and all_few_shot_results[g][var_name] is not None:
                    needs_base = False
                needs_optuna = True
                if loaded_fs_yolo and all_few_shot_results[g].get(var_name_optuna) is not None:
                    needs_optuna = False
            else:
                if loaded_fs and all_few_shot_results[g][var_name] is not None:
                    needs_base = False

            if not needs_base and not needs_optuna:
                continue

            try:
                if needs_base:
                    print(f"Evaluating Few-Shot: {var_name} - {g}")
                    evaluator = Nutrition5kEvaluator(
                        pred_json_path=pred_path,
                        nlp_model=nlp_model,
                        vocabulary=vocabulary,
                        vocab_embeddings=vocab_embeddings,
                        threshold=threshold
                    )
                    nlp_res = evaluator.evaluate_nlp_metrics()
                    macro_res = evaluator.evaluate_macronutrients()
                    
                    all_few_shot_results[g][var_name] = {
                        "nlp": nlp_res,
                        "macro": macro_res
                    }
                
                if needs_optuna:
                    print(f"Evaluating Few-Shot: {var_name_optuna} - {g}")
                    optuna_summary_path = os.path.join("data", "optuna", "few_shot_yolo_image", folder, "optuna_summary.json")
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
                        evaluator_opt = Nutrition5kEvaluator(
                            pred_json_path=pred_path,
                            nlp_model=nlp_model,
                            vocabulary=vocabulary,
                            vocab_embeddings=vocab_embeddings,
                            threshold=threshold,
                            optuna_scalars=optuna_scalars
                        )
                        all_few_shot_results[g][var_name_optuna] = {
                            "nlp": evaluator_opt.evaluate_nlp_metrics(),
                            "macro": evaluator_opt.evaluate_macronutrients()
                        }
                    else:
                        print(f"[!] Could not find optuna scalars for {filename} in {optuna_summary_path}")

            except Exception as e:
                print(f"[!] Error in {pred_path}: {e}")

    def save_fs_csv(path, is_yolo):
        print(f"Saving {path}...")
        with open(path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['Group', 'Variant']
            for std_key in ["Non-Standardized", "Standardized"]:
                for k in ["P", "R", "F1", "IoU"]:
                    fieldnames.append(f"nlp_{std_key}_{k}")
            for m_key in ["overall", "exact_match"]:
                for f_name in ["calories", "mass", "fat", "carbohydrates", "protein"]:
                    for metric in ["MAE", "MAE_%"]:
                        fieldnames.append(f"macro_{m_key}_{f_name}_{metric}")
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for grp in GROUPS:
                for var in FEW_SHOT_VARIANTS:
                    if ("YOLO" in var) != is_yolo:
                        continue
                    res = all_few_shot_results[grp][var]
                    if not res:
                        continue
                    row = {'Group': grp, 'Variant': var}
                    for std_key in ["Non-Standardized", "Standardized"]:
                        for k in ["P", "R", "F1", "IoU"]:
                            row[f"nlp_{std_key}_{k}"] = res["nlp"][std_key][k]
                    for m_key in ["overall", "exact_match"]:
                        for f_name in ["calories", "mass", "fat", "carbohydrates", "protein"]:
                            for metric in ["MAE", "MAE_%"]:
                                row[f"macro_{m_key}_{f_name}_{metric}"] = res["macro"][m_key][f"{f_name}_{metric}"]
                    writer.writerow(row)

    if updated_fs:
        save_fs_csv(fs_csv_path, False)
    if updated_fs_yolo:
        save_fs_csv(fs_yolo_csv_path, True)
                
    # Generate Output Tables
    print("\n" + "=" * 120)
    print(" " * 45 + "FEW-SHOT SUMMARY TABLE")
    print("=" * 120)
    
    # NLP Table
    print("\n--- FEW-SHOT NLP METRICS ---")
    header_nlp = f"{'Group':<12} | {'Variant':<25} | {'F1 (Non-Std)':<12} | {'F1 (Std)':<10} | {'IoU (Non-Std)':<13} | {'IoU (Std)':<10}"
    print("-" * len(header_nlp))
    print(header_nlp)
    print("-" * len(header_nlp))
    for g in GROUPS:
        for v in FEW_SHOT_VARIANTS:
            res = all_few_shot_results[g][v]
            if res and "nlp" in res and "Standardized" in res["nlp"]:
                f1_s = res["nlp"]["Standardized"]["F1"]
                iou_s = res["nlp"]["Standardized"]["IoU"]
                f1_ns = res["nlp"]["Non-Standardized"]["F1"]
                iou_ns = res["nlp"]["Non-Standardized"]["IoU"]
                print(f"{g:<12} | {v:<25} | {f1_ns:<12.4f} | {f1_s:<10.4f} | {iou_ns:<13.4f} | {iou_s:<10.4f}")
            else:
                print(f"{g:<12} | {v:<25} | {'N/A':<12} | {'N/A':<10} | {'N/A':<13} | {'N/A':<10}")
    print("-" * len(header_nlp))
    
    # Macro Table (Overall)
    print("\n--- FEW-SHOT MACRO METRICS (Overall MAE & MAE %) ---")
    header_macro = f"{'Group':<12} | {'Variant':<25} | {'Calories':<18} | {'Mass':<18} | {'Fat':<18} | {'Carbs':<18} | {'Protein':<18}"
    print("-" * len(header_macro))
    print(header_macro)
    print("-" * len(header_macro))
    for g in GROUPS:
        for v in FEW_SHOT_VARIANTS:
            res = all_few_shot_results[g][v]
            if res and "macro" in res and "overall" in res["macro"]:
                overall = res["macro"]["overall"]
                cal = f"{overall.get('calories_MAE', 0.0):.1f} ({overall.get('calories_MAE_%', 0.0):.1f}%)"
                mass = f"{overall.get('mass_MAE', 0.0):.1f} ({overall.get('mass_MAE_%', 0.0):.1f}%)"
                fat = f"{overall.get('fat_MAE', 0.0):.1f} ({overall.get('fat_MAE_%', 0.0):.1f}%)"
                carbs = f"{overall.get('carbohydrates_MAE', 0.0):.1f} ({overall.get('carbohydrates_MAE_%', 0.0):.1f}%)"
                prot = f"{overall.get('protein_MAE', 0.0):.1f} ({overall.get('protein_MAE_%', 0.0):.1f}%)"
                print(f"{g:<12} | {v:<25} | {cal:<18} | {mass:<18} | {fat:<18} | {carbs:<18} | {prot:<18}")
            else:
                print(f"{g:<12} | {v:<25} | {'N/A':<18} | {'N/A':<18} | {'N/A':<18} | {'N/A':<18} | {'N/A':<18}")
    print("-" * len(header_macro))

    # Macro Table (Exact Match)
    print("\n--- FEW-SHOT MACRO METRICS (Exact Match MAE & MAE %) ---")
    print("-" * len(header_macro))
    print(header_macro)
    print("-" * len(header_macro))
    for g in GROUPS:
        for v in FEW_SHOT_VARIANTS:
            res = all_few_shot_results[g][v]
            if res and "macro" in res and "exact_match" in res["macro"]:
                exact = res["macro"]["exact_match"]
                cal = f"{exact.get('calories_MAE', 0.0):.1f} ({exact.get('calories_MAE_%', 0.0):.1f}%)"
                mass = f"{exact.get('mass_MAE', 0.0):.1f} ({exact.get('mass_MAE_%', 0.0):.1f}%)"
                fat = f"{exact.get('fat_MAE', 0.0):.1f} ({exact.get('fat_MAE_%', 0.0):.1f}%)"
                carbs = f"{exact.get('carbohydrates_MAE', 0.0):.1f} ({exact.get('carbohydrates_MAE_%', 0.0):.1f}%)"
                prot = f"{exact.get('protein_MAE', 0.0):.1f} ({exact.get('protein_MAE_%', 0.0):.1f}%)"
                print(f"{g:<12} | {v:<25} | {cal:<18} | {mass:<18} | {fat:<18} | {carbs:<18} | {prot:<18}")
            else:
                print(f"{g:<12} | {v:<25} | {'N/A':<18} | {'N/A':<18} | {'N/A':<18} | {'N/A':<18} | {'N/A':<18}")
    print("-" * len(header_macro))
    
    # Plotting
    output_dir = os.path.join("data/evaluation", "few_shot_metrics")
    plot_few_shot_metrics(all_few_shot_results, output_dir)
    print(f"Few-shot charts saved in {output_dir}")



if __name__ == "__main__":
    init_dynamic_globals()

    # ================= CONFIGURATIONS =================
    THRESHOLD = 0.6
    OUTPUT_DIR = "data/evaluation"

    SIMILARITY_DIR = os.path.join(OUTPUT_DIR, "similarity_analysis")
    THRESHOLD_DIR = os.path.join(OUTPUT_DIR, "threshold_study")
    NLP_DIR = os.path.join(OUTPUT_DIR, "nlp_metrics")
    MAE_OVERALL_DIR = os.path.join(OUTPUT_DIR, "mae_metrics", "overall")
    MAE_EXACT_DIR = os.path.join(OUTPUT_DIR, "mae_metrics", "exact_match")
    MAE_OVERLAY_DIR = os.path.join(OUTPUT_DIR, "mae_metrics", "overlay")

    for d in [SIMILARITY_DIR, THRESHOLD_DIR, NLP_DIR, MAE_OVERALL_DIR, MAE_EXACT_DIR, MAE_OVERLAY_DIR]:
        os.makedirs(d, exist_ok=True)

    PRED_JSON_PATHS = [
        "data/prediction/one_example/test_predictions_nutrition5k_gemma4_e2b_q4_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_gemma4_e2b_q8_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_twostep_gemma4_e2b_q4_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_twostep_gemma4_e2b_q8_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_hybrid_twostep_gemma4_e2b_q4_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_hybrid_twostep_gemma4_e2b_q8_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_v2hybrid_twostep_gemma4_e2b_q4_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_v2hybrid_twostep_gemma4_e2b_q8_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_gemma4_e4b_q4_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_gemma4_e4b_q8_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_twostep_gemma4_e4b_q4_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_twostep_gemma4_e4b_q8_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_hybrid_twostep_gemma4_e4b_q4_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_hybrid_twostep_gemma4_e4b_q8_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_v2hybrid_twostep_gemma4_e4b_q4_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_v2hybrid_twostep_gemma4_e4b_q8_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_yolo_gemma4_e2b_q4_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_yolo_gemma4_e2b_q8_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_yolo_gemma4_e4b_q4_tb.json",
        "data/prediction/one_example/test_predictions_nutrition5k_yolo_gemma4_e4b_q8_tb.json"
    ]

    INGREDIENTS_CSV_PATH = "data/metadata/ingredients_metadata.csv"
    # ==================================================
    print("Loading vocabulary from CSV...")
    vocabulary = load_vocabulary(INGREDIENTS_CSV_PATH)

    print("Loading NLP model: all-MiniLM-L6-v2... (Happens only once)")
    nlp_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Calculating embeddings for the base vocabulary...")
    vocab_embeddings = nlp_model.encode(vocabulary)

    all_nlp_results = {}
    all_macro_results = {}
    eval_csv_path = os.path.join(OUTPUT_DIR, "nlp_evaluation.csv")
    
    if os.path.exists(eval_csv_path):
        print(f"Loading existing evaluation results from {eval_csv_path}...")
        with open(eval_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                m_name = row['model_name']
                all_nlp_results[m_name] = {
                    "Non-Standardized": {k: float(row[f"nlp_Non-Standardized_{k}"]) for k in ["P", "R", "F1", "IoU"]},
                    "Standardized": {k: float(row[f"nlp_Standardized_{k}"]) for k in ["P", "R", "F1", "IoU"]}
                }
                all_macro_results[m_name] = {
                    "overall": {},
                    "exact_match": {}
                }
                for f_name in ["calories", "mass", "fat", "carbohydrates", "protein"]:
                    all_macro_results[m_name]["overall"][f"{f_name}_MAE"] = float(row[f"macro_overall_{f_name}_MAE"])
                    all_macro_results[m_name]["overall"][f"{f_name}_MAE_%"] = float(row[f"macro_overall_{f_name}_MAE_%"])
                    all_macro_results[m_name]["exact_match"][f"{f_name}_MAE"] = float(row[f"macro_exact_match_{f_name}_MAE"])
                    all_macro_results[m_name]["exact_match"][f"{f_name}_MAE_%"] = float(row[f"macro_exact_match_{f_name}_MAE_%"])

    updated_evals = False

    for pred_path in PRED_JSON_PATHS:
        model_name = os.path.basename(pred_path).replace('test_predictions_nutrition5k_', '').replace('.json', '')

        is_yolo = "yolo" in model_name.lower()
        model_name_optuna = model_name + "_optuna"

        needs_base = model_name not in all_nlp_results or model_name not in all_macro_results
        needs_optuna = is_yolo and (
                    model_name_optuna not in all_nlp_results or model_name_optuna not in all_macro_results)

        if not needs_base and not needs_optuna:
            continue

        print(f"\n{'=' * 70}")
        print(f"MODEL EVALUATION: {model_name}")
        print(f"{'=' * 70}")
        updated_evals = True

        try:
            if needs_base:
                evaluator = Nutrition5kEvaluator(
                    pred_json_path=pred_path,
                    nlp_model=nlp_model,
                    vocabulary=vocabulary,
                    vocab_embeddings=vocab_embeddings,
                    threshold=THRESHOLD
                )

                nlp_res = evaluator.evaluate_nlp_metrics()
                all_nlp_results[model_name] = nlp_res

                # Export and print similarity results for the model
                sim_path = os.path.join(SIMILARITY_DIR, f"similarity_analysis_{model_name}.json")
                evaluator.export_similarity_log_to_json(sim_path)
                evaluator.print_similarity_summary()

                macro_res = evaluator.evaluate_macronutrients()
                all_macro_results[model_name] = macro_res

            if needs_optuna:
                print(f"--- Evaluating YOLO Optuna variant: {model_name_optuna} ---")
                optuna_summary_path = os.path.join("data", "optuna", "few_shot_yolo_image", "one_example", "optuna_summary.json")
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
                    evaluator_opt = Nutrition5kEvaluator(
                        pred_json_path=pred_path,
                        nlp_model=nlp_model,
                        vocabulary=vocabulary,
                        vocab_embeddings=vocab_embeddings,
                        threshold=THRESHOLD,
                        optuna_scalars=optuna_scalars
                    )
                    all_nlp_results[model_name_optuna] = evaluator_opt.evaluate_nlp_metrics()
                    all_macro_results[model_name_optuna] = evaluator_opt.evaluate_macronutrients()
                else:
                    print(f"[!] Could not find optuna scalars for {model_name} in {optuna_summary_path}")

        except Exception as e:
            print(f"\n[!] Error during the execution of file {pred_path}:\n{e}")

    if updated_evals:
        print(f"Saving evaluation results to {eval_csv_path}...")
        with open(eval_csv_path, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['model_name']
            for std_key in ["Non-Standardized", "Standardized"]:
                for k in ["P", "R", "F1", "IoU"]:
                    fieldnames.append(f"nlp_{std_key}_{k}")
            for m_key in ["overall", "exact_match"]:
                for f_name in ["calories", "mass", "fat", "carbohydrates", "protein"]:
                    fieldnames.append(f"macro_{m_key}_{f_name}_MAE")
                    fieldnames.append(f"macro_{m_key}_{f_name}_MAE_%")
            
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
                    for f_name in ["calories", "mass", "fat", "carbohydrates", "protein"]:
                        row[f"macro_{m_key}_{f_name}_MAE"] = all_macro_results[m_name][m_key][f"{f_name}_MAE"]
                        row[f"macro_{m_key}_{f_name}_MAE_%"] = all_macro_results[m_name][m_key][f"{f_name}_MAE_%"]
                writer.writerow(row)

    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETED. GENERATING GRAPHS...")
    print(f"{'=' * 70}")

    if all_nlp_results:
        plot_nlp_metrics(all_nlp_results, output_dir=NLP_DIR)
    if all_macro_results:
        overall_results = {k: v["overall"] for k, v in all_macro_results.items()}
        exact_results = {k: v["exact_match"] for k, v in all_macro_results.items()}
        plot_macro_metrics(overall_results, output_dir=MAE_OVERALL_DIR, plot_baseline=True)
        plot_macro_metrics(exact_results, output_dir=MAE_EXACT_DIR, plot_baseline=False)
        plot_overlay_macro_metrics(all_macro_results, output_dir=MAE_OVERLAY_DIR)

    print("\n" + "=" * 70)
    print("PRINTING SUMMARY TABLE TO TERMINAL...")
    print("=" * 70)
    print_summary_tables(all_nlp_results, all_macro_results)

    print(f"\n{'=' * 70}")
    print("STARTING THRESHOLD STUDY FOR NLP METRICS...")
    print(f"{'=' * 70}")

    thresholds = list(np.linspace(1.0, 0.0, 11))
    study_json_path = os.path.join(THRESHOLD_DIR, "threshold_study_results.json")

    if os.path.exists(study_json_path):
        print(f"Loading existing threshold study from {study_json_path}...")
        with open(study_json_path, 'r', encoding='utf-8') as f:
            threshold_study_results = json.load(f)
    else:
        threshold_study_results = {}

    updated = False
    for pred_path in PRED_JSON_PATHS:
        model_name = os.path.basename(pred_path).replace('test_predictions_nutrition5k_', '').replace('.json', '')
        if model_name in threshold_study_results:
            continue

        print(f"Computing threshold study for missing model: {model_name}...")
        updated = True
        threshold_study_results[model_name] = {'F1': [], 'IoU': []}

        try:
            evaluator = Nutrition5kEvaluator(
                pred_json_path=pred_path,
                nlp_model=nlp_model,
                vocabulary=vocabulary,
                vocab_embeddings=vocab_embeddings,
                threshold=1.0
            )

            for thr in thresholds:
                evaluator.threshold = thr
                print(f"======== EVALUATING THRESHOLD {thr:.2f} ========")
                nlp_res = evaluator.evaluate_nlp_metrics()
                if nlp_res and "Standardized" in nlp_res:
                    threshold_study_results[model_name]['F1'].append(nlp_res["Standardized"]["F1"])
                    threshold_study_results[model_name]['IoU'].append(nlp_res["Standardized"]["IoU"])
                else:
                    threshold_study_results[model_name]['F1'].append(0.0)
                    threshold_study_results[model_name]['IoU'].append(0.0)
        except Exception as e:
            print(f"\n[!] Error during the threshold study of file {pred_path}:\n{e}")
            if model_name in threshold_study_results:
                del threshold_study_results[model_name]

    if updated:
        print("Saving updated threshold study results...")
        with open(study_json_path, 'w', encoding='utf-8') as f:
            json.dump(threshold_study_results, f, indent=4)

    plot_threshold_study(thresholds, threshold_study_results, output_dir=THRESHOLD_DIR)

    # === FEW SHOT EVALUATION ===
    evaluate_few_shot_learning(nlp_model, vocabulary, vocab_embeddings, THRESHOLD)

    # === YOLO OPTUNA COMPARISON ===
    def plot_yolo_optuna_metrics(results: dict, output_dir: str):
        import matplotlib.lines as mlines
        from matplotlib.ticker import ScalarFormatter

        os.makedirs(output_dir, exist_ok=True)

        fields = ["calories", "mass", "fat", "carbohydrates", "protein"]

        baseline = {
            "calories_MAE": 70.6, "calories_MAE_%": 26.1,
            "mass_MAE": 40.4, "mass_MAE_%": 18.8,
            "fat_MAE": 5.0, "fat_MAE_%": 34.2,
            "carbohydrates_MAE": 6.1, "carbohydrates_MAE_%": 31.9,
            "protein_MAE": 5.5, "protein_MAE_%": 29.5
        }

        # 1. Definizione Simboli e Colori (YOLO vs YOLO Optuna)
        strategies = ['YOLO', 'YOLO Optuna']
        examples = ['1 Example', '5 Examples', '10 Examples']

        strat_colors = {
            'YOLO': '#ff9933',  # Arancione
            'YOLO Optuna': '#cc0000'  # Rosso Scuro
        }
        strat_markers = {
            'YOLO': 'P',  # Croce riempita (Plus)
            'YOLO Optuna': 'X'  # X maiuscola
        }

        # Offset leggermente più vicini visto che ci sono solo 2 strategie per colonna
        ex_offsets = {'1 Example': -0.15, '5 Examples': 0.0, '10 Examples': 0.15}

        def plot_yolo_macro_type(metric_suffix, file_suffix, ylabel, title_prefix, subset="overall", plot_baseline=True,
                                 log_scale=False):
            out_dir = os.path.join(output_dir, subset)
            os.makedirs(out_dir, exist_ok=True)

            fig_macro, axes_macro = plt.subplots(2, 3, figsize=(18, 12))
            axes_macro = axes_macro.flatten()

            def draw_dot_plot(ax, f):
                # Linea di Baseline
                if plot_baseline:
                    base_val = baseline[f"{f}_{metric_suffix}"]
                    ax.axhline(y=base_val, color='gray', linestyle='--', linewidth=1.2, zorder=1)

                    text_y = base_val * 1.05 if log_scale else base_val
                    ax.text(x=-0.45, y=text_y, s=' Nutrition 5k ', color='#333333', fontsize=10,
                            va='center', fontweight='bold', zorder=5,
                            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2))

                for grp_idx, g in enumerate(GROUPS):
                    if grp_idx > 0:
                        ax.axvline(x=grp_idx - 0.5, color='#e0e0e0', linestyle='-', linewidth=1, zorder=0)

                    for ex_name in examples:
                        x_pos = grp_idx + ex_offsets[ex_name]

                        for strat in strategies:
                            var_name = f"{ex_name} ({strat})"
                            res = results.get(g, {}).get(var_name)

                            val = 0.0
                            if res and "macro" in res and subset in res["macro"]:
                                val = res["macro"][subset][f"{f}_{metric_suffix}"]

                            val = max(val, 0.01) if log_scale else val

                            if val > 0.01 or not log_scale:
                                is_below = plot_baseline and (val < base_val)
                                edgecolor = 'black' if is_below else 'white'
                                lw = 1.5 if is_below else 0.8

                                ax.scatter(x_pos, val, color=strat_colors[strat], marker=strat_markers[strat],
                                           s=110, edgecolors=edgecolor, linewidth=lw, alpha=0.85, zorder=4)

                ax.set_ylabel(ylabel, fontsize=12)
                if log_scale:
                    ax.set_yscale('log')
                    ax.yaxis.set_major_formatter(ScalarFormatter())

                ax.set_title(f'{f.capitalize()}', fontsize=14, fontweight='bold')
                ax.set_xticks(np.arange(len(GROUPS)))
                ax.set_xticklabels(GROUPS, fontsize=12, fontweight='bold')
                ax.tick_params(axis='y', labelsize=11)
                ax.grid(False, axis='x')
                ax.grid(True, axis='y', linestyle=':', color='#cccccc', zorder=0)
                ax.set_xlim(-0.5, len(GROUPS) - 0.5)

            # Combina i 5 grafici
            for idx, f in enumerate(fields):
                draw_dot_plot(axes_macro[idx], f)

            # Sesto Riquadro per la Legenda
            axes_macro[5].axis('off')
            legend_elements = []

            legend_elements.append(mlines.Line2D([], [], color='none', label='STRATEGIA (Forma e Colore):'))
            for strat in strategies:
                legend_elements.append(mlines.Line2D([], [], color='none', marker=strat_markers[strat],
                                                     markerfacecolor=strat_colors[strat], markeredgecolor='white',
                                                     markersize=10, label=f"  {strat}"))

            legend_elements.append(mlines.Line2D([], [], color='none', label=''))
            legend_elements.append(mlines.Line2D([], [], color='none', label='N. ESEMPI (Posizione su asse X):'))
            legend_elements.append(
                mlines.Line2D([], [], color='none', marker='|', markeredgecolor='black', markersize=10,
                              label='  Sinistra = 1 Esempio'))
            legend_elements.append(
                mlines.Line2D([], [], color='none', marker='|', markeredgecolor='black', markersize=10,
                              label='  Centro = 5 Esempi'))
            legend_elements.append(
                mlines.Line2D([], [], color='none', marker='|', markeredgecolor='black', markersize=10,
                              label='  Destra = 10 Esempi'))

            if plot_baseline:
                legend_elements.append(mlines.Line2D([], [], color='none', label=''))
                legend_elements.append(mlines.Line2D([], [], color='none', marker='o', markerfacecolor='#e0e0e0',
                                                     markeredgecolor='black', linewidth=1.5, markersize=12,
                                                     label='Bordo nero = Migliore del Baseline'))

            axes_macro[5].legend(handles=legend_elements, loc='center left', fontsize=11, frameon=False, borderpad=1)

            title_suffix = " (Log Scale)" if log_scale else ""
            plt.suptitle(f"YOLO Optuna Comparison - {title_prefix}{title_suffix}", fontsize=18, fontweight='bold',
                         y=0.98)
            plt.tight_layout(rect=[0, 0.02, 1, 0.96])

            file_suffix_out = f"{file_suffix}_log" if log_scale else file_suffix
            plt.savefig(os.path.join(out_dir, f'yolo_optuna_{subset}_{file_suffix_out}_combined.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()

            # Disegna i grafici singoli
            for f in fields:
                fig_single, ax_single = plt.subplots(figsize=(10, 8))
                draw_dot_plot(ax_single, f)

                single_leg_elements = []
                for strat in strategies:
                    single_leg_elements.append(mlines.Line2D([], [], color='none', marker=strat_markers[strat],
                                                             markerfacecolor=strat_colors[strat],
                                                             markeredgecolor='white',
                                                             markersize=10, label=strat))

                ax_single.legend(handles=single_leg_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                                 ncol=2, fontsize=11, title="Strategie (Sinistra=1 ex, Centro=5 ex, Destra=10 ex)",
                                 title_fontsize=12, frameon=False)

                plt.tight_layout(rect=[0, 0.15, 1, 1])
                plt.savefig(os.path.join(out_dir, f'yolo_optuna_{subset}_{file_suffix_out}_{f}.png'),
                            bbox_inches='tight', dpi=300)
                plt.close()

        # Richiama la funzione per generare tutte le varianti (lineari e logaritmiche)
        plot_yolo_macro_type("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Overall)", "overall",
                             plot_baseline=True, log_scale=False)
        plot_yolo_macro_type("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Overall)", "overall",
                             plot_baseline=True, log_scale=False)
        plot_yolo_macro_type("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Exact Match)",
                             "exact_match", plot_baseline=False, log_scale=False)
        plot_yolo_macro_type("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Exact Match)",
                             "exact_match", plot_baseline=False, log_scale=False)

        plot_yolo_macro_type("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Overall)", "overall",
                             plot_baseline=True, log_scale=True)
        plot_yolo_macro_type("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Overall)", "overall",
                             plot_baseline=True, log_scale=True)
        plot_yolo_macro_type("MAE", "mae_abs", "Mean Absolute Error (Value)", "Absolute MAE (Exact Match)",
                             "exact_match", plot_baseline=False, log_scale=True)
        plot_yolo_macro_type("MAE_%", "mae_perc", "Mean Absolute Error (%)", "Percentage MAE (Exact Match)",
                             "exact_match", plot_baseline=False, log_scale=True)

    def evaluate_yolo_optuna_comparison(nlp_model, vocabulary, vocab_embeddings, threshold, base_dir="data/prediction"):
        print("\n" + "=" * 70)
        print("STARTING YOLO OPTUNA EVALUATION COMPARISON")
        print("=" * 70)
        
        folders = ["one_example", "five_example", "ten_example"]
        folder_to_prefix = {"one_example": "1 Example", "five_example": "5 Examples", "ten_example": "10 Examples"}
        
        results = {}
        yolo_optuna_json_path = "data/evaluation/few_shot_yolo_optuna_evaluation.json"

        if os.path.exists(yolo_optuna_json_path):
            print(f"Loading existing YOLO Optuna evaluation from {yolo_optuna_json_path}")
            with open(yolo_optuna_json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        
        updated = False
        
        for folder in folders:
            pred_folder = os.path.join(base_dir, folder)
            optuna_summary_path = os.path.join("data", "optuna", "few_shot_yolo_image", folder, "optuna_summary.json")
            
            if not os.path.exists(pred_folder) or not os.path.exists(optuna_summary_path):
                continue
                
            with open(optuna_summary_path, 'r', encoding='utf-8') as f:
                optuna_data = json.load(f)
                
            for filename in os.listdir(pred_folder):
                if not filename.endswith('.json') or 'yolo' not in filename.lower():
                    continue
                    
                g, v_type = parse_model_name(filename)
                if g not in GROUPS: continue
                
                prefix = folder_to_prefix[folder]
                var_name = f"{prefix} (YOLO)"
                var_name_optuna = f"{prefix} (YOLO Optuna)"
                
                if g not in results:
                    results[g] = {}
                    
                if var_name_optuna in results[g] and var_name in results[g]:
                    continue
                    
                updated = True
                
                # Extract the base model identifier to match the optuna json key
                # filename: test_predictions_nutrition5k_yolo_gemma4_e2b_q4_tb.json
                # optuna key: optuna_train_predictions_gemma4_e2b_q4_tb.json
                model_identifier = filename.replace('test_predictions_nutrition5k_yolo_', '').replace('.json', '')
                
                optuna_scalars = None
                for key, val in optuna_data.items():
                    if model_identifier in key:
                        optuna_scalars = val.get("optimal_scalars")
                        break
                        
                if not optuna_scalars:
                    print(f"[!] Could not find optuna scalars for {filename} in {optuna_summary_path}")
                    continue
                    
                print(f"Evaluating YOLO Optuna: {var_name_optuna} - {g}")
                pred_path = os.path.join(pred_folder, filename)
                
                # Evaluate WITHOUT optuna (if not already in results)
                if var_name not in results[g]:
                    evaluator_base = Nutrition5kEvaluator(pred_path, nlp_model, vocabulary, vocab_embeddings, threshold)
                    results[g][var_name] = {
                        "nlp": evaluator_base.evaluate_nlp_metrics(),
                        "macro": evaluator_base.evaluate_macronutrients()
                    }
                    
                # Evaluate WITH optuna
                evaluator_opt = Nutrition5kEvaluator(pred_path, nlp_model, vocabulary, vocab_embeddings, threshold, optuna_scalars=optuna_scalars)
                results[g][var_name_optuna] = {
                    "nlp": evaluator_opt.evaluate_nlp_metrics(),
                    "macro": evaluator_opt.evaluate_macronutrients()
                }
                
        if updated:
            with open(yolo_optuna_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
                
        # Now plot the comparison graphs...
        plot_yolo_optuna_metrics(results, output_dir="data/evaluation/yolo_optuna_metrics")
        print(f"YOLO Optuna charts saved in data/evaluation/yolo_optuna_metrics")

    evaluate_yolo_optuna_comparison(nlp_model, vocabulary, vocab_embeddings, THRESHOLD)


    print(f"\nAll graphs (single and combined) have been saved in the folder: {OUTPUT_DIR}")