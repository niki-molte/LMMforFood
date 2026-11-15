import argparse
import glob
import json
import os
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer, util


class EvaluationLookupTable:
    def __init__(
        self,
        metadata_csv_path: str,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.6
    ):
        """
        Initialize the Semantic Matching evaluator.

        Args:
            metadata_csv_path: Path to the ingredients_metadata.csv file
            model_name: sentence-transformers model used to compute embeddings
            similarity_threshold: Minimum cosine similarity required to accept an ingredient match
        """
        self.similarity_threshold = similarity_threshold

        # 1. Load the metadata CSV file
        print(f"Loading ingredient vocabulary from: {metadata_csv_path}")
        self.metadata_df = pd.read_csv(metadata_csv_path)

        # Check that the expected columns are present
        expected_cols = ['ingr', 'id', 'cal/g', 'fat(g)', 'carb(g)', 'protein(g)']
        for col in expected_cols:
            if col not in self.metadata_df.columns:
                raise ValueError(f"Missing column in CSV: {col}")

        # Remove duplicate ingredient names, keeping the first occurrence
        self.metadata_df = self.metadata_df.drop_duplicates(subset=['ingr'], keep='first')

        self.ingredients_vocab = self.metadata_df['ingr'].tolist()

        # Lookup dictionary for O(1) access to per-gram macronutrients
        self.lookup_dict = self.metadata_df.set_index('ingr').to_dict('index')

        # 2. Set up the Semantic Matching model
        print(f"Initializing SentenceTransformer model: {model_name}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)

        print("Generating embeddings for the ingredient vocabulary...")
        self.vocab_embeddings = self.model.encode(self.ingredients_vocab, convert_to_tensor=True)
        print("Initialization complete.\n")

    def find_best_match(self, predicted_ingredient: str) -> Tuple[Optional[str], float]:
        """
        Compute the cosine similarity between the predicted ingredient and the CSV vocabulary.

        Returns:
            Tuple of (matched_ingredient_name, similarity_score).
            If the score is below the threshold, matched_ingredient_name will be None.
        """
        if not predicted_ingredient or not isinstance(predicted_ingredient, str):
            return None, 0.0

        pred_embedding = self.model.encode(predicted_ingredient, convert_to_tensor=True)

        # Compute cosine similarity against all vocabulary embeddings
        cosine_scores = util.cos_sim(pred_embedding, self.vocab_embeddings)[0]

        # Find the best match
        best_idx = torch.argmax(cosine_scores).item()
        best_score = cosine_scores[best_idx].item()
        best_match_ingr = self.ingredients_vocab[best_idx]

        if best_score >= self.similarity_threshold:
            return best_match_ingr, best_score
        else:
            return None, best_score

    def extract_predicted_ingredients(self, dish_pred: Dict) -> List[Dict]:
        """
        Extract the list of predicted ingredients from a JSON dictionary.
        Dynamically handles several possible JSON structures.
        """
        items = []

        # Pattern 1a: Two-Step/Hybrid structure with "predicted_nutrition_step2"
        # (takes precedence over predicted_nutrition since it is the pipeline's final result)
        if 'predicted_nutrition_step2' in dish_pred:
            nut_pred = dish_pred['predicted_nutrition_step2']
            if 'visible_ingredients' in nut_pred:
                items.extend(nut_pred['visible_ingredients'])
            if 'hidden_ingredients' in nut_pred:
                items.extend(nut_pred['hidden_ingredients'])

        # Pattern 1b: Structure with "predicted_nutrition" (Standard/YOLO)
        elif 'predicted_nutrition' in dish_pred:
            nut_pred = dish_pred['predicted_nutrition']
            if 'visible_ingredients' in nut_pred:
                items.extend(nut_pred['visible_ingredients'])
            if 'hidden_ingredients' in nut_pred:
                items.extend(nut_pred['hidden_ingredients'])

        # Pattern 2: Flat lists with generic keys
        elif 'ingredients' in dish_pred:
            items.extend(dish_pred['ingredients'])
        elif 'predicted_ingredients' in dish_pred:
            items.extend(dish_pred['predicted_ingredients'])

        return items

    def process_dish(self, dish_pred: Dict) -> Dict:
        """
        Process the prediction for a single dish:
        - Extract ingredients and weights
        - Validate each ingredient via Semantic Matching
        - Compute the total macronutrients from the matched per-gram values
        """
        ingredients_list = self.extract_predicted_ingredients(dish_pred)

        # Accumulators for the dish totals
        total_cal = 0.0
        total_fat = 0.0
        total_carb = 0.0
        total_prot = 0.0

        validated_ingredients = []

        for item in ingredients_list:
            if not isinstance(item, dict):
                continue

            # Check keys for the ingredient name ("ingredient" or "name") and weight ("weight_grams")
            pred_name = item.get('ingredient') or item.get('name')
            weight_grams = item.get('weight_grams')

            if not pred_name or weight_grams is None:
                continue

            try:
                weight_grams = float(weight_grams)
            except (ValueError, TypeError):
                continue

            # STEP 1: Semantic Matching
            best_match_ingr, score = self.find_best_match(pred_name)

            ingr_result = {
                "original_prediction": pred_name,
                "weight_grams": weight_grams,
                "similarity_score": round(score, 4)
            }

            if best_match_ingr:
                # Valid match (>= threshold): retrieve the values from the CSV
                macros_per_g = self.lookup_dict[best_match_ingr]

                # STEP 2: Compute macronutrients for the ingredient
                cal = weight_grams * macros_per_g['cal/g']
                fat = weight_grams * macros_per_g['fat(g)']
                carb = weight_grams * macros_per_g['carb(g)']
                prot = weight_grams * macros_per_g['protein(g)']

                # Update the dish totals
                total_cal += cal
                total_fat += fat
                total_carb += carb
                total_prot += prot

                ingr_result.update({
                    "matched_ingredient": best_match_ingr,
                    "status": "accepted",
                    "matched_macronutrients": {
                        "calories": round(cal, 4),
                        "fat": round(fat, 4),
                        "carbohydrates": round(carb, 4),
                        "protein": round(prot, 4)
                    }
                })
            else:
                # Below threshold: fall back to the macronutrients originally predicted by the LLM
                orig_macros = item.get('macronutrients', {})
                if not isinstance(orig_macros, dict):
                    orig_macros = {}

                try:
                    cal = float(orig_macros.get('calories', 0.0) or 0.0)
                    fat = float(orig_macros.get('fat', 0.0) or 0.0)
                    carb = float(orig_macros.get('carbohydrates', 0.0) or 0.0)
                    prot = float(orig_macros.get('protein', 0.0) or 0.0)
                except (ValueError, TypeError):
                    cal = fat = carb = prot = 0.0

                # Update the dish totals with the original fallback values
                total_cal += cal
                total_fat += fat
                total_carb += carb
                total_prot += prot

                ingr_result.update({
                    "matched_ingredient": None,
                    "status": "unmatched",
                    "matched_macronutrients": {
                        "calories": round(cal, 4),
                        "fat": round(fat, 4),
                        "carbohydrates": round(carb, 4),
                        "protein": round(prot, 4)
                    }
                })

            validated_ingredients.append(ingr_result)

        # Add the post-processed results to the dish's original dictionary
        dish_pred['semantic_validated_ingredients'] = validated_ingredients
        dish_pred['semantic_matching_total_macronutrients'] = {
            "calories": round(total_cal, 4),
            "fat": round(total_fat, 4),
            "carbohydrates": round(total_carb, 4),
            "protein": round(total_prot, 4)
        }

        return dish_pred

    def process_file(self, input_json_path: str, output_json_path: str) -> List[Dict]:
        """
        Read a JSON file of predictions, process each dish, and save the updated output.
        """
        print(f"Processing predictions from: {input_json_path}")
        with open(input_json_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)

        # Handle both single JSON objects and lists of objects
        is_single_object = isinstance(predictions, dict)
        if is_single_object:
            predictions = [predictions]

        processed_data = []
        for dish_pred in predictions:
            processed_dish = self.process_dish(dish_pred)
            processed_data.append(processed_dish)

        if is_single_object:
            processed_data = processed_data[0]

        # Save the output
        os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4)

        print(f"Processing complete! File saved to: {output_json_path}")
        return processed_data

    def process_directory(self, input_dir: str, output_dir: str):
        """
        Process all JSON files in a directory and save them to the output directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        json_files = glob.glob(os.path.join(input_dir, "*.json"))

        print(f"Found {len(json_files)} files in {input_dir}")
        for file_path in json_files:
            filename = os.path.basename(file_path)
            out_path = os.path.join(output_dir, filename)
            self.process_file(file_path, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Matching post-processing for LMM predictions.")
    parser.add_argument("--metadata", type=str, required=True,
                        help="Path to the ingredients_metadata.csv file")
    parser.add_argument("--input", type=str, required=False,
                        help="Path to a single JSON file containing the predictions")
    parser.add_argument("--output", type=str, required=False,
                        help="Path where the processed single JSON file will be saved")
    parser.add_argument("--batch_dir", type=str, required=False,
                        help="Path to the prediction folder (e.g. datasets/Nutrition5k/data/prediction); "
                             "every subfolder is processed into a matching _semantic folder")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Cosine similarity threshold (default: 0.6)")

    args = parser.parse_args()

    evaluator = EvaluationLookupTable(
        metadata_csv_path=args.metadata,
        similarity_threshold=args.threshold
    )

    if args.batch_dir:
        print(f"\nStarting BATCH processing for base folder: {args.batch_dir}")
        for item in os.listdir(args.batch_dir):
            input_dir = os.path.join(args.batch_dir, item)

            # Skip single files and folders that are already "_semantic"
            if not os.path.isdir(input_dir) or item.endswith("_semantic"):
                continue

            output_dir = os.path.join(args.batch_dir, f"{item}_semantic")
            print(f"\n--- Processing subfolder: {item} -> {item}_semantic ---")
            evaluator.process_directory(input_dir, output_dir)

        print("\nBATCH processing completed successfully!")

    elif args.input and args.output:
        evaluator.process_file(args.input, args.output)
    else:
        print("Error: you must specify either (--input and --output) or --batch_dir.")


# python evaluation_lookup_table.py --metadata data/metadata/ingredients_metadata.csv --batch_dir data/optuna/few_shot_standard