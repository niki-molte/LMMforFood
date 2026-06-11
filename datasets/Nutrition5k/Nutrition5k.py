class LoadNutrition5kDataset(BaseFoodDatasetLoader):
    """
    Dataset loader specific to the Nutrition5k format.
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
        """
        Parses CSV metadata and builds the record dictionaries including multi-angle videos.
        """
        print("Starting metadata processing...")
        for csv_path in self.dish_metadata_paths:
            if not os.path.exists(csv_path):
                print(f"WARNING: CSV file not found -> {csv_path}")
                continue

            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue

                    dish_id = row[0]
                    split = self.get_split_for_item(dish_id)

                    if not split:
                        continue

                    try:
                        num_ingrs = int(float(row[6]))
                    except ValueError:
                        continue

                    ingredient_data = row[7:]
                    ingredients = []

                    if num_ingrs > 0 and len(ingredient_data) > 0:
                        step = len(ingredient_data) // num_ingrs
                        for i in range(num_ingrs):
                            ingr_name_idx = i * step + 1
                            if ingr_name_idx < len(ingredient_data):
                                ingredients.append(ingredient_data[ingr_name_idx])

                    record = {
                        "image_path": f"imagery/realsense_overhead/{dish_id}/rgb.png",
                        "depth_path": f"imagery/realsense_overhead/{dish_id}/depth.png",
                        "food_id": dish_id,
                        "food_category": "Nutrition5k Plate",
                        "ingredients": ingredients,
                        "side_angles_videos": [
                            f"imagery/side_angles/{dish_id}/camera_{cam}.h264"
                            for cam in ['A', 'B', 'C', 'D']
                        ]
                    }

                    self.dataset_records[split].append(record)

            print(f"Successfully processed file {csv_path}")


# ======= USAGE EXAMPLE =======
if __name__ == "__main__":
    OUTPUT_DIRECTORY = "data/"

    # Nutrition5k Processing
    NUTRITION_DIR = "/percorso/del/tuo/dataset/Nutrition5k"
    nutrition_loader = LoadNutrition5kDataset(NUTRITION_DIR, OUTPUT_DIRECTORY)
    nutrition_loader.load_metadata()
    nutrition_loader.process()
    nutrition_loader.save_json()