import os
import json
from collections import defaultdict


class BaseFoodDatasetLoader:
    """
    Base class containing common logic for parsing dataset splits and saving to JSON.
    """

    def __init__(self, dataset_dir, output_dir):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.splits = defaultdict(set)
        self.dataset_records = defaultdict(list)

    def load_splits_from_files(self, split_paths_dict):
        """
        Reads text files containing IDs/paths and populates the splits dictionary.
        """
        for split_name, filepath in split_paths_dict.items():
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        item_id = line.strip()
                        if item_id:
                            self.splits[split_name].add(item_id)
            else:
                print(f"WARNING: Split file not found -> {filepath}")

    def get_split_for_item(self, item_id):
        """
        Checks all loaded splits and returns the split name to which the item belongs.
        """
        for split_name, items_set in self.splits.items():
            if item_id in items_set:
                return split_name
        return None

    def save_json(self):
        """
        Exports the processed records into separate JSON files based on their split.
        """
        print("Saving JSON files...")
        os.makedirs(self.output_dir, exist_ok=True)

        for split_name, records in self.dataset_records.items():
            if records:
                output_file = os.path.join(self.output_dir, f"{split_name}_dataset.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(records, f, ensure_ascii=False, indent=4)
                print(f"Saved {split_name}_dataset.json with {len(records)} items.")