import os
import json


class LoadDataset:
    def __init__(self, dataset_dir, output_dir):
        """
        Initializes the class with dataset and output directory paths.
        """
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir

        # File paths
        self.food_list_path = os.path.join(dataset_dir, 'FoodList.txt')
        self.ingredient_list_path = os.path.join(dataset_dir, 'IngredientList.txt')
        self.ingre_label_path = os.path.join(dataset_dir, 'IngreLabel.txt')

        # Split paths
        self.split_paths = {
            'train': os.path.join(dataset_dir, 'TR.txt'),
            'val': os.path.join(dataset_dir, 'VAL.txt'),
            'test': os.path.join(dataset_dir, 'TE.txt')
        }

        # In-memory data structures
        self.foods = {}
        self.ingredients = {}
        self.splits = {'train': set(), 'val': set(), 'test': set()}

    def load_metadata(self):
        """Loads food names, ingredient names, and dataset splits."""
        print("Loading metadata...")

        # 1. Load food names (1-indexed based on the folder name)
        with open(self.food_list_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                self.foods[i + 1] = line.strip()

        # 2. Load ingredient names (0-indexed based on the column)
        with open(self.ingredient_list_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                self.ingredients[i] = line.strip()

        # 3. Load splits and save them into Sets for O(1) lookup
        for split_name, filepath in self.split_paths.items():
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Ensure paths match exactly (strip extra whitespaces)
                        img_path = line.strip()
                        self.splits[split_name].add(img_path)
            else:
                print(f"Warning: Split file not found -> {filepath}")

    def get_split_for_image(self, image_path):
        """Determines which split (train/val/test) an image belongs to."""
        if image_path in self.splits['train']: return 'train'
        if image_path in self.splits['val']: return 'val'
        if image_path in self.splits['test']: return 'test'
        return None

    def process_and_save(self):
        """Processes the large IngreLabel.txt file and saves the results as JSON."""
        print("Starting to process IngreLabel.txt...")
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize containers for JSON data
        dataset_records = {'train': [], 'val': [], 'test': []}

        with open(self.ingre_label_path, 'r', encoding='utf-8') as f:
            for line_count, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue

                image_path = parts[0]
                labels = parts[1:]

                # Find the split for the current image
                split = self.get_split_for_image(image_path)
                if not split:
                    # Skip if the image is not tracked in TR, TE, or VAL files
                    continue

                # Extract food ID from the path (e.g., "/100/xiachufang_1.jpg" -> 100)
                try:
                    clean_path = image_path.lstrip('/')
                    food_id = int(clean_path.split('/')[0])
                    food_name = self.foods.get(food_id, "Unknown")
                except (ValueError, IndexError):
                    food_id = -1
                    food_name = "Unknown"

                # Extract present ingredients (label '1')
                present_ingredients = []
                for idx, label in enumerate(labels):
                    if label == '1':
                        present_ingredients.append(self.ingredients.get(idx, f"Unknown_Ing_{idx}"))

                # Create the record dictionary
                record = {
                    "image_path": image_path,
                    "food_id": food_id,
                    "food_category": food_name,
                    "ingredients": present_ingredients
                }

                dataset_records[split].append(record)

                # Print status every 10,000 lines
                if line_count % 10000 == 0:
                    print(f"Processed {line_count} images...")

        # Save JSON files for LLM experiments
        print("Saving JSON files...")
        for split, records in dataset_records.items():
            output_file = os.path.join(self.output_dir, f"{split}_dataset.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=4)
            print(f"Saved {split}_dataset.json with {len(records)} dishes.")


# ======= USAGE EXAMPLE =======
if __name__ == "__main__":
    # Ensure the dataset .txt files are located in this directory
    DATASET_DIRECTORY = "/media/niki/Niki My Passport/scuola/università/magistrale/tesi/dataset/VireoFood172/SplitAndIngreLabel/SplitAndIngreLabel"

    # Directory where the generated JSONs will be saved
    OUTPUT_DIRECTORY = "data/"

    loader = LoadDataset(DATASET_DIRECTORY, OUTPUT_DIRECTORY)
    loader.load_metadata()
    loader.process_and_save()