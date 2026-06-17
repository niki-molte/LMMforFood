import json
import os
import csv
import statistics
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Nutrition5kEvaluator:
    def __init__(self, pred_json_path: str, ingredients_csv_path: str, model_name: str = 'all-MiniLM-L6-v2',
                 threshold: float = 0.8):
        self.pred_path = pred_json_path
        self.ingredients_csv_path = ingredients_csv_path
        self.threshold = threshold

        self.predictions = self._load_json(self.pred_path)

        print("Caricamento del vocabolario da CSV...")
        self.vocabulary = self._load_vocabulary(self.ingredients_csv_path)

        print(f"Caricamento del modello NLP: {model_name}...")
        self.nlp_model = SentenceTransformer(model_name)

        print("Calcolo degli embedding per il vocabolario base (potrebbe richiedere qualche secondo)...")
        self.vocab_embeddings = self.nlp_model.encode(self.vocabulary)

        self.similarity_log = []

    def _load_json(self, path: str) -> list:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File JSON non trovato: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_vocabulary(self, path: str) -> list:
        """Legge la colonna 'ingr' dal file ingredients_metadata.csv."""
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

    def _clean_ingredients(self, ingredients: list) -> list:
        """Estrae e pulisce il nome dell'ingrediente dalla lista di dizionari."""
        return [str(ing.get("name", "")).strip().lower()
                for ing in ingredients
                if isinstance(ing, dict) and str(ing.get("name", "")).strip()]

    def _standardize_list(self, pred_list: list, recipe_id: str = "N/A") -> set:
        """Standardizza gli ingredienti predetti confrontandoli con l'intero vocabolario CSV."""
        if not pred_list:
            return set()

        pred_embeddings = self.nlp_model.encode(pred_list)
        # Confrontiamo le predizioni con TUTTO il vocabolario CSV
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
        print("\nCalcolo metriche NLP (P, R, F1, IoU)...")
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

            # nlp non standardizzate
            m_exact = self._compute_set_metrics(pred_set_exact, gt_set)
            for k in sums["non_std"]: sums["non_std"][k] += m_exact[k]

            # nlp standardizzate
            pred_set_std = self._standardize_list(pred_list, recipe_id)
            m_std = self._compute_set_metrics(pred_set_std, gt_set)
            for k in sums["std"]: sums["std"][k] += m_std[k]

            valid_items += 1

        if valid_items == 0:
            return {}

        return {
            "Non-Standardized": {k: v / valid_items for k, v in sums["non_std"].items()},
            "Standardized (NLP)": {k: v / valid_items for k, v in sums["std"].items()}
        }

    def evaluate_macronutrients(self) -> dict:
        print("\nCalcolo statistiche Macronutrienti (MAE e MAE%)...")
        DATA_FIELDNAMES = ["calories", "mass", "fat", "carbohydrates", "protein"]

        groundtruth_values = {field: [] for field in DATA_FIELDNAMES}
        err_values = {field: [] for field in DATA_FIELDNAMES}

        for item in self.predictions:

            # calcolo ground truth
            gt_cal = gt_mass = gt_fat = gt_carbs = gt_prot = 0.0
            for ing in item.get("ground_truth_ingredients", []):
                gt_mass += ing.get("weight_grams", 0.0)
                macros = ing.get("macronutrients", {})
                gt_cal += macros.get("calories", 0.0)
                gt_fat += macros.get("fat", 0.0)
                gt_carbs += macros.get("carbohydrates", 0.0)
                gt_prot += macros.get("protein", 0.0)

            # calcolo dei macro
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


if __name__ == "__main__":

    PRED_JSON_PATH = "data/prediction/test_predictions_nutrition5k_gemma4e2b.json"
    #PRED_JSON_PATH = "data/prediction/perfect_predictions.json"
    INGREDIENTS_CSV_PATH = "/media/niki/Niki My Passport/scuola/università/magistrale/tesi/dataset/Nutrition5k/nutrition5k_dataset/metadata/ingredients_metadata.csv"

    try:
        evaluator = Nutrition5kEvaluator(
            pred_json_path=PRED_JSON_PATH,
            ingredients_csv_path=INGREDIENTS_CSV_PATH,
            threshold=0.8
        )

        nlp_results = evaluator.evaluate_nlp_metrics()
        for mode, metrics in nlp_results.items():
            print(f"\n--- NLP {mode} Metrics ---")
            for m_name, val in metrics.items():
                print(f"{m_name}: {val:.4f}")

        macro_results = evaluator.evaluate_macronutrients()
        print("\n--- Macronutrients Error (MAE & MAE%) ---")
        for m_name, val in macro_results.items():
            print(f"{m_name}: {val:.2f}")

    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")