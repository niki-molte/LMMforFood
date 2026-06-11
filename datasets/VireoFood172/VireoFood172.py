from datasets.BaseDataset import BaseFoodDatasetLoader
import os

class LoadVireoFood172(BaseFoodDatasetLoader):
    """
    Dataset loader specific to the VireoFood172 format.
    """

    def __init__(self, dataset_dir, output_dir):
        super().__init__(dataset_dir, output_dir)
        self.food_list_path = os.path.join(dataset_dir, 'FoodList.txt')
        self.ingredient_list_path = os.path.join(dataset_dir, 'IngredientList.txt')
        self.ingredients_label_path = os.path.join(dataset_dir, 'IngreLabel.txt')

        self.split_paths = {
            'train': os.path.join(dataset_dir, 'TR.txt'),
            'val': os.path.join(dataset_dir, 'VAL.txt'),
            'test': os.path.join(dataset_dir, 'TE.txt')
        }

        self.foods = {}
        self.ingredients = {}

    def load_metadata(self):
        """
        Loads dataset-specific foods, ingredients, and text-based splits.
        """
        print("Loading VireoFood172 metadata...")
        with open(self.food_list_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                self.foods[i + 1] = line.strip()

        with open(self.ingredient_list_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                self.ingredients[i] = line.strip()

        self.load_splits_from_files(self.split_paths)

    def process(self):
        """
        Parses IngreLabel.txt and builds the record dictionaries.
        """
        print("Starting to process IngreLabel.txt...")
        with open(self.ingredients_label_path, 'r', encoding='utf-8') as f:
            for line_count, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue

                image_path = parts[0]
                labels = parts[1:]
                split = self.get_split_for_item(image_path)

                if not split:
                    continue

                try:
                    clean_path = image_path.lstrip('/')
                    food_id = int(clean_path.split('/')[0])
                    food_name = self.foods.get(food_id, "Unknown")
                except (ValueError, IndexError):
                    food_id = -1
                    food_name = "Unknown"

                present_ingredients = [
                    self.ingredients.get(idx, f"Unknown_Ing_{idx}")
                    for idx, label in enumerate(labels) if label == '1'
                ]

                record = {
                    "image_path": image_path,
                    "food_id": food_id,
                    "food_category": food_name,
                    "ingredients": present_ingredients
                }

                self.dataset_records[split].append(record)

                if line_count % 10000 == 0:
                    print(f"Processed {line_count} images...")

if __name__ == "__main__":
    OUTPUT_DIRECTORY = "data/"

    # VireoFood172 Processing
    VIREO_DIR = "/media/niki/Niki My Passport/scuola/università/magistrale/tesi/dataset/VireoFood172/SplitAndIngreLabel"
    vireo_loader = LoadVireoFood172(VIREO_DIR, OUTPUT_DIRECTORY)
    vireo_loader.load_metadata()
    vireo_loader.process()
    vireo_loader.save_json()

    print("-" * 40)