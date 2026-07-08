import os
import csv

from datasets.BaseDataset import BaseFoodDatasetLoader


class LoadNutrition5kItalianoDataset(BaseFoodDatasetLoader):
    """
    Dataset loader specific to the Nutrition5k format.
    Extracts complete macronutrient profiles for both the total dish and individual ingredients.
    """

    def __init__(self, dataset_dir, output_dir):
        super().__init__(dataset_dir, output_dir)
        self.dish_metadata_paths = [
            os.path.join(dataset_dir, 'metadata', 'Clean_DB1_ITAcorr_FINALE_REP3 1(Sheet).csv'),
            os.path.join(dataset_dir, 'metadata', 'Clean_DB2_ITAcorr_FINALE_REP3(Sheet).csv')
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

            if not os.path.exists(csv_path):
                print(f"WARNING: CSV file not found -> {csv_path}")
                continue

            count = 0
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=';')

                current_dish_id = None
                current_split = None
                record = None

                for row in reader:
                    if not row:
                        continue

                    # skip first rows.
                    if count <= 2:
                        count += 1
                        continue

                    raw_dish_id = row[0].strip()

                    # new dish found
                    if raw_dish_id:

                        # save previous dish
                        if record and current_split:
                            image_full_path = os.path.join(self.dataset_dir, record["image_path"])
                            if not os.path.exists(image_full_path):
                                print(f"WARNING: immagine non trovata su disco per {current_dish_id}, record scartato.")
                            else:
                                self.dataset_records[current_split].append(record)

                        # dish initialization
                        current_dish_id = raw_dish_id
                        current_split = self.get_split_for_item(current_dish_id)

                        if not current_split:
                            # if dish is not in split no computation
                            record = None
                            continue

                        record = {
                            "image_path": f"imagery/realsense_overhead/{current_dish_id}/rgb.png",
                            "depth_path": f"imagery/realsense_overhead/{current_dish_id}/depth_color.png",
                            "food_id": current_dish_id,
                            "food_category": "Nutrition5k Plate",
                            "total_macronutrients": {
                                "calories": 0.0,
                                "total_mass_grams": 0.0,
                                "fat": 0.0,
                                "carbohydrates": 0.0,
                                "protein": 0.0
                            },
                            "ingredients": [],
                            "side_angles_videos": [
                                f"imagery/side_angles/{current_dish_id}/camera_{cam}.h264"
                                for cam in ['A', 'B', 'C', 'D']
                            ]
                        }

                    # dish is not in split
                    if not record:
                        continue

                    # add current ingredient
                    try:
                        ingr_name = row[2]
                        ingr_grams = float(row[3].replace(',', '.'))

                        ingr_cal = float(row[92].replace(',', '.'))
                        ingr_prot = float(row[94].replace(',', '.'))
                        ingr_fat = float(row[97].replace(',', '.'))
                        ingr_carb = float(row[101].replace(',', '.'))

                        record["ingredients"].append({
                            "name": ingr_name,
                            "weight_grams": ingr_grams,
                            "macronutrients": {
                                "calories": ingr_cal,
                                "fat": ingr_fat,
                                "carbohydrates": ingr_carb,
                                "protein": ingr_prot
                            }
                        })

                        record["total_macronutrients"]["calories"] += ingr_cal
                        record["total_macronutrients"]["total_mass_grams"] += ingr_grams
                        record["total_macronutrients"]["fat"] += ingr_fat
                        record["total_macronutrients"]["carbohydrates"] += ingr_carb
                        record["total_macronutrients"]["protein"] += ingr_prot

                    except ValueError:
                        print(f"WARNING: Invalid ingredient data for dish {current_dish_id} (row: {row[2]}).")

                # save the record
                if record and current_split:
                    image_full_path = os.path.join(self.dataset_dir, record["image_path"])
                    if not os.path.exists(image_full_path):
                        print(f"WARNING: immagine non trovata su disco per {current_dish_id}, record scartato.")
                    else:
                        self.dataset_records[current_split].append(record)

            print(f"Successfully processed file {csv_path}")


# ======= USAGE EXAMPLE =======
if __name__ == "__main__":
    OUTPUT_DIRECTORY = "data/ground truth/"
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Nutrition5k Processing
    NUTRITION_DIR = "/media/niki/Niki My Passport/scuola/università/magistrale/tesi/dataset/Nutrition5k/nutrition5k_dataset"
    nutrition_loader = LoadNutrition5kItalianoDataset(NUTRITION_DIR, OUTPUT_DIRECTORY)
    nutrition_loader.load_metadata()

    nutrition_loader.process()
    nutrition_loader.save_json()