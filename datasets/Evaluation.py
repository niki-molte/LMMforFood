import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EvaluatePredictions:
    def __init__(self, pred_path: str, model_name: str = 'all-MiniLM-L6-v2', threshold: float = 0.8):
        self.pred_path = pred_path
        self.threshold = threshold
        self.predictions = self._load_json(self.pred_path)

        print(f"Caricamento del modello NLP: {model_name}...")
        self.nlp_model = SentenceTransformer(model_name)

        self.similarity_log = []

    def _load_json(self, path: str) -> list:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _clean_ingredients(self, ingredients: list) -> list:
        return [str(ing).strip().lower() for ing in ingredients if str(ing).strip()]

    def _standardize_list(self, pred_list: list, gt_list: list, recipe_id: str = "N/A") -> set:
        if not pred_list or not gt_list:
            return set(pred_list)

        pred_embeddings = self.nlp_model.encode(pred_list)
        gt_embeddings = self.nlp_model.encode(gt_list)
        sim_matrix = cosine_similarity(pred_embeddings, gt_embeddings)

        standardized_preds = set()

        for i, pred_ing in enumerate(pred_list):
            best_match_idx = np.argmax(sim_matrix[i])
            best_score = float(sim_matrix[i][best_match_idx])

            accepted = best_score >= self.threshold
            best_gt_match = gt_list[best_match_idx]

            self.similarity_log.append({
                "recipe_id": recipe_id,
                "predicted": pred_ing,
                "best_gt_match": best_gt_match,
                "max_similarity_score": round(best_score, 4),
                "threshold_passed": accepted
            })

            if accepted:
                standardized_preds.add(best_gt_match)
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

    def calculate_all_metrics(self) -> dict:
        sums = {
            "non_std": {"P": 0.0, "R": 0.0, "F1": 0.0, "IoU": 0.0},
            "std": {"P": 0.0, "R": 0.0, "F1": 0.0, "IoU": 0.0}
        }
        valid_items = 0

        self.similarity_log = []

        for item in self.predictions:
            gt_raw = item.get("ground_truth_ingredients", [])
            pred_dict = item.get("predicted_ingredients", {})
            pred_raw = pred_dict.get("visible_ingredients", []) + pred_dict.get("hidden_ingredients", [])

            recipe_id = item.get("image_path", "unknown")

            gt_list = self._clean_ingredients(gt_raw)
            pred_list = self._clean_ingredients(pred_raw)

            if not gt_list:
                continue

            gt_set = set(gt_list)
            pred_set_exact = set(pred_list)

            m_exact = self._compute_set_metrics(pred_set_exact, gt_set)
            for k in sums["non_std"]: sums["non_std"][k] += m_exact[k]

            pred_set_std = self._standardize_list(pred_list, gt_list, recipe_id)
            m_std = self._compute_set_metrics(pred_set_std, gt_set)
            for k in sums["std"]: sums["std"][k] += m_std[k]

            valid_items += 1

        if valid_items == 0:
            return {}

        return {
            "Non-Standardized": {k: v / valid_items for k, v in sums["non_std"].items()},
            "Standardized (NLP)": {k: v / valid_items for k, v in sums["std"].items()}
        }

    def export_similarity_log_to_json(self, filename: str = "similarity_analysis.json"):
        """
        Exports the similarity log to a structured JSON file grouped by recipe_id.
        """
        if not self.similarity_log:
            print("Nessun log da esportare. Esegui prima calculate_all_metrics().")
            return

        json_output = {}
        for log in self.similarity_log:
            rid = log["recipe_id"]
            if rid not in json_output:
                json_output[rid] = {}

            json_output[rid][log["predicted"]] = {
                "best_gt_match": log["best_gt_match"],
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

        print("\n--- Riassunto Analisi Similarità NLP ---")
        print(f"Totale confronti effettuati: {total}")
        print(f"Ingredienti standardizzati (sopra soglia {self.threshold}): {passed} ({(passed / total) * 100:.1f}%)")
        print(f"Ingredienti rimasti invariati: {total - passed} ({((total - passed) / total) * 100:.1f}%)")
        print(f"Score di similarità massima media (su tutti i sample): {avg_score:.4f}")


if __name__ == "__main__":
    base_dir = "VireoFood172/data/"
    pred_file = os.path.join(base_dir, "prediction/test_predictions.json")

    try:
        evaluator = EvaluatePredictions(pred_file, threshold=0.8)
        all_results = evaluator.calculate_all_metrics()

        for mode, metrics in all_results.items():
            print(f"\n--- {mode} Metrics ---")
            for m_name, val in metrics.items():
                print(f"{m_name}: {val:.4f}")

        evaluator.print_similarity_summary()
        evaluator.export_similarity_log_to_json(os.path.join(base_dir, "prediction/similarity_analysis.json"))

    except FileNotFoundError as e:
        print(f"Errore: {e}. Verifica i percorsi dei file.")