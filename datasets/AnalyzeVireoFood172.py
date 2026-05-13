import os
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

class AnalyzeDataset:

    def __init__(self, train_path: str, val_path: str, test_path: str):
        if train_path is not None and os.path.exists(train_path):
            self.train_data = pd.read_json(train_path)
        else:
            raise FileNotFoundError("The training dataset JSON does not exist.")

        if val_path is not None and os.path.exists(val_path):
            self.val_data = pd.read_json(val_path)
        else:
            raise FileNotFoundError("The validation dataset JSON does not exist.")

        if test_path is not None and os.path.exists(test_path):
            self.test_data = pd.read_json(test_path)
        else:
            raise FileNotFoundError("The testing dataset JSON does not exist.")

        self.concatenated_dataset = None  # Inizializziamo a None

    def get_dataframes(self):
        return self.train_data, self.val_data, self.test_data

    def get_concatenated_dataframe(self):
        train_df = self.train_data.copy()
        val_df = self.val_data.copy()
        test_df = self.test_data.copy()

        train_df["dataset_split"] = "train"
        val_df["dataset_split"] = "val"
        test_df["dataset_split"] = "test"

        self.concatenated_dataset = pd.concat([train_df, val_df, test_df], ignore_index=True)

        expected_len = len(train_df) + len(val_df) + len(test_df)
        if len(self.concatenated_dataset) != expected_len:
            print("ATTENTION! some data are lost.")

        return self.concatenated_dataset

if __name__ == "__main__":
    TRAIN_PATH = "VireoFood172/data/ground truth/train_dataset.json"
    VAL_PATH = "VireoFood172/data/ground truth/val_dataset.json"
    TEST_PATH = "VireoFood172/data/ground truth/test_dataset.json"

    VireoFood = AnalyzeDataset(TRAIN_PATH, VAL_PATH, TEST_PATH)
    print(VireoFood.get_concatenated_dataframe())
