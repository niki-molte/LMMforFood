"""
evaluator.py
============
Evaluation class (NLP + macronutrients) for a single run of predictions.
Evaluation logic for a single predictions file.
"""
import json
import os
import statistics
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Nutrition5kEvaluator:
    def __init__(self, pred_json_path: str, nlp_model, vocabulary: list, vocab_embeddings, threshold: float,
                 optuna_scalars: dict = None):
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

            if "predicted_nutrition_step2" in item:
                pred_dict = item.get("predicted_nutrition_step2", {})
            else:
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
            for k in sums["non_std"]:
                sums["non_std"][k] += m_exact[k]

            pred_set_std = self._standardize_list(pred_list, recipe_id)
            m_std = self._compute_set_metrics(pred_set_std, gt_set)
            for k in sums["std"]:
                sums["std"][k] += m_std[k]

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

            if "semantic_matching_validated_ingredients" in item:
                predicted_ingredients = item.get("semantic_matching_validated_ingredients", [])
            elif "rag_validated_ingredients" in item:
                predicted_ingredients = item.get("rag_validated_ingredients", [])
            else:
                if "predicted_nutrition_step2" in item:
                    pred_nut = item.get("predicted_nutrition_step2", {})
                else:
                    pred_nut = item.get("predicted_nutrition", {})
                predicted_ingredients = pred_nut.get("visible_ingredients", []) + pred_nut.get("hidden_ingredients", [])

            def get_ingr_name(ing):
                return str(
                    ing.get("name", ing.get("matched_ingredient", ing.get("original_prediction", "")))).strip().lower()

            pred_list = [get_ingr_name(ing) for ing in predicted_ingredients if get_ingr_name(ing)]
            pred_set = set(pred_list)

            exact_names = gt_set.intersection(pred_set)

            gt_cal = gt_mass = gt_fat = gt_carbs = gt_prot = 0.0
            gt_cal_ex = gt_mass_ex = gt_fat_ex = gt_carbs_ex = gt_prot_ex = 0.0

            for ing in gt_raw:
                name = str(ing.get("name", "")).strip().lower()
                mass = ing.get("weight_grams", 0.0)
                macros = ing.get("macronutrients", {})
                if not isinstance(macros, dict):
                    macros = {}
                cal = macros.get("calories", 0.0)
                fat = macros.get("fat", 0.0)
                carbs = macros.get("carbohydrates", 0.0)
                prot = macros.get("protein", 0.0)

                gt_mass += mass
                gt_cal += cal
                gt_fat += fat
                gt_carbs += carbs
                gt_prot += prot
                if name in exact_names:
                    gt_mass_ex += mass
                    gt_cal_ex += cal
                    gt_fat_ex += fat
                    gt_carbs_ex += carbs
                    gt_prot_ex += prot

            pred_cal = pred_mass = pred_fat = pred_carbs = pred_prot = 0.0
            pred_cal_ex = pred_mass_ex = pred_fat_ex = pred_carbs_ex = pred_prot_ex = 0.0

            for ing in predicted_ingredients:
                name = get_ingr_name(ing)
                mass = ing.get("weight_grams", 0.0)

                if "deterministic_macronutrients" in ing and ing["deterministic_macronutrients"] is not None:
                    macros = ing["deterministic_macronutrients"]
                else:
                    macros = ing.get("macronutrients", {})

                if not isinstance(macros, dict):
                    macros = {}
                cal = macros.get("calories", 0.0)
                fat = macros.get("fat", 0.0)
                carbs = macros.get("carbohydrates", 0.0)
                prot = macros.get("protein", 0.0)

                pred_mass += mass
                pred_cal += cal
                pred_fat += fat
                pred_carbs += carbs
                pred_prot += prot
                if name in exact_names:
                    pred_mass_ex += mass
                    pred_cal_ex += cal
                    pred_fat_ex += fat
                    pred_carbs_ex += carbs
                    pred_prot_ex += prot

            gt_vals = {"calories": gt_cal, "mass": gt_mass, "fat": gt_fat, "carbohydrates": gt_carbs,
                       "protein": gt_prot}
            pred_vals = {"calories": pred_cal, "mass": pred_mass, "fat": pred_fat, "carbohydrates": pred_carbs,
                         "protein": pred_prot}

            gt_vals_ex = {"calories": gt_cal_ex, "mass": gt_mass_ex, "fat": gt_fat_ex, "carbohydrates": gt_carbs_ex,
                          "protein": gt_prot_ex}
            pred_vals_ex = {"calories": pred_cal_ex, "mass": pred_mass_ex, "fat": pred_fat_ex,
                            "carbohydrates": pred_carbs_ex, "protein": pred_prot_ex}

            if self.optuna_scalars:
                def _piecewise(val, nut_key):
                    thr = self.optuna_scalars.get(f"threshold_{nut_key}", 0.0)
                    s_small = self.optuna_scalars.get(f"scale_small_{nut_key}", 1.0)
                    s_large = self.optuna_scalars.get(f"scale_large_{nut_key}", 1.0)
                    return max(0.0, val * s_small if val < thr else val * s_large)

                is_piecewise = "threshold_mass" in self.optuna_scalars
                scale_mass = self.optuna_scalars.get("scale_mass", 1.0)

                is_mass_only = list(self.optuna_scalars.keys()) == ["scale_mass"]
                _fallback = scale_mass if is_mass_only else 1.0

                scale_cal = self.optuna_scalars.get("scale_cal", _fallback)
                scale_fat = self.optuna_scalars.get("scale_fat", _fallback)
                scale_carbs = self.optuna_scalars.get("scale_carbs", _fallback)
                scale_protein = self.optuna_scalars.get("scale_protein", _fallback)

                if is_piecewise:
                    pred_vals["mass"] = _piecewise(pred_vals["mass"], "mass")
                    pred_vals["calories"] = _piecewise(pred_vals["calories"], "cal")
                    pred_vals["fat"] = _piecewise(pred_vals["fat"], "fat")
                    pred_vals["carbohydrates"] = _piecewise(pred_vals["carbohydrates"], "carbs")
                    pred_vals["protein"] = _piecewise(pred_vals["protein"], "protein")

                    pred_vals_ex["mass"] = _piecewise(pred_vals_ex["mass"], "mass")
                    pred_vals_ex["calories"] = _piecewise(pred_vals_ex["calories"], "cal")
                    pred_vals_ex["fat"] = _piecewise(pred_vals_ex["fat"], "fat")
                    pred_vals_ex["carbohydrates"] = _piecewise(pred_vals_ex["carbohydrates"], "carbs")
                    pred_vals_ex["protein"] = _piecewise(pred_vals_ex["protein"], "protein")
                else:
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
            mean_err = statistics.mean(err_values[field])
            mean_gt = statistics.mean(groundtruth_values[field])
            output_stats["overall"][f"{field}_MAE"] = mean_err
            output_stats["overall"][f"{field}_MAE_%"] = (100 * mean_err / mean_gt) if mean_gt > 0 else 0.0

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
