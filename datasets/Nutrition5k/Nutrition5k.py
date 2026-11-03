import os
import csv

from datasets.BaseDataset import BaseFoodDatasetLoader
from settings import get_env, get_path  # centralized configuration (.env)


class LoadNutrition5kDataset(BaseFoodDatasetLoader):
    """
    Dataset loader specific to the Nutrition5k format.
    Extracts complete macronutrient profiles for both the total dish and individual ingredients.
    """

    def __init__(self, dataset_dir, output_dir):
        super().__init__(dataset_dir, output_dir)
        self.dish_metadata_paths = [
            os.path.join(dataset_dir, 'metadata', 'dish_metadata_cafe1.csv'),
            os.path.join(dataset_dir, 'metadata', 'dish_metadata_cafe2.csv')
        ]

        self.split_paths = {
            'train': os.path.join(dataset_dir, 'dish_ids', 'splits', 'rgb_train_ids.txt'),
            'test': os.path.join(dataset_dir, 'dish_ids', 'splits', 'rgb_test_ids.txt')
        }

    def load_metadata(self):
        """
        Loads dataset-specific text-based splits.
        """
        print("Loading Nutrition5k metadata...")
        self.load_splits_from_files(self.split_paths)

    def process(self):
        print("Starting metadata processing...")
        for csv_path in self.dish_metadata_paths:

            # check if metadata CSV exists
            if not os.path.exists(csv_path):
                print(f"WARNING: CSV file not found -> {csv_path}")
                continue

            # start reading CSV metadata files
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue

                    dish_id = row[0].strip()
                    split = self.get_split_for_item(dish_id)

                    if not split:
                        print(f"INFO: dish_id '{dish_id}' not found in any split, skipping.")
                        continue

                    try:
                        total_calories = float(row[1])
                        total_mass_grams = float(row[2])
                        total_fat = float(row[3])
                        total_carbohydrates = float(row[4])
                        total_protein = float(row[5])
                    except ValueError:
                        print(f"WARNING: Invalid numeric data for dish {dish_id}. Skipping.")
                        continue

                    ingredient_data = row[6:]
                    ingredients = []
                    step = 7

                    i = 0
                    while i + step <= len(ingredient_data):
                        base_idx = i
                        try:
                            ingr_id = ingredient_data[base_idx + 0]
                            ingr_name = ingredient_data[base_idx + 1]
                            ingr_grams = float(ingredient_data[base_idx + 2])
                            ingr_cal = float(ingredient_data[base_idx + 3])
                            ingr_fat = float(ingredient_data[base_idx + 4])
                            ingr_carb = float(ingredient_data[base_idx + 5])
                            ingr_prot = float(ingredient_data[base_idx + 6])

                            ingredients.append({
                                "name": ingr_name,
                                "weight_grams": ingr_grams,
                                "macronutrients": {
                                    "calories": ingr_cal,
                                    "fat": ingr_fat,
                                    "carbohydrates": ingr_carb,
                                    "protein": ingr_prot
                                }
                            })
                        except ValueError:
                            print(f"WARNING: Invalid ingredient data at block {i // step} for dish {dish_id}.")
                        i += step

                    residuo = len(ingredient_data) % step
                    if residuo != 0:
                        print(f"WARNING: dish {dish_id} has {residuo} trailing ingredient field(s) ignored.")

                    record = {
                        "image_path": f"imagery/realsense_overhead/{dish_id}/rgb.png",
                        "depth_path": f"imagery/realsense_overhead/{dish_id}/depth_color.png",
                        "food_id": dish_id,
                        "food_category": "Nutrition5k Plate",
                        "total_macronutrients": {
                            "calories": total_calories,
                            "total_mass_grams": total_mass_grams,
                            "fat": total_fat,
                            "carbohydrates": total_carbohydrates,
                            "protein": total_protein
                        },
                        "ingredients": ingredients,
                        "side_angles_videos": [
                            f"imagery/side_angles/{dish_id}/camera_{cam}.h264"
                            for cam in ['A', 'B', 'C', 'D']
                        ]
                    }

                    image_full_path = os.path.join(self.dataset_dir, record["image_path"])
                    if not os.path.exists(image_full_path):
                        print(f"WARNING: image not found on disk for {dish_id}, record discarded.")
                        continue

                    self.dataset_records[split].append(record)

            print(f"Successfully processed file {csv_path}")


# ======= USAGE EXAMPLE =======
if __name__ == "__main__":
    OUTPUT_DIRECTORY = "data/ground truth/"
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Nutrition5k Processing
    NUTRITION_DIR = get_path("NUTRITION5K_DATASET_DIR")
    nutrition_loader = LoadNutrition5kDataset(NUTRITION_DIR, OUTPUT_DIRECTORY)
    nutrition_loader.load_metadata()

    nutrition_loader.process()
    #nutrition_loader.save_json()