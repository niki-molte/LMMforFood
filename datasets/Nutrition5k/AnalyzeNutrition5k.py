import os
import json
import pandas as pd
import numpy as np
import scipy as sp
import itertools
from collections import Counter
import networkx as nx
import matplotlib

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class AnalyzeNutrition5k:
    """
    A class used to load, analyze, and visualize distributions, complexities,
    and out-of-vocabulary challenges of the Nutrition5k dataset.
    """

    def __init__(self, train_path: str, test_path: str):
        """
        Initializes the AnalyzeNutrition5k instance by loading JSON data files.
        """
        self.train_data = self._load_json(train_path, "training")
        self.test_data = self._load_json(test_path, "testing")

        self.concatenated_dataset = None
        self.exploded_ingredients = None

    def _load_json(self, path: str, dataset_name: str) -> pd.DataFrame:
        """
        Private helper to validate and load a JSON file into a DataFrame.
        """
        if path is not None and os.path.exists(path):
            return pd.read_json(path)
        raise FileNotFoundError(f"The Nutrition5k JSON dataset for {dataset_name} does not exist at: {path}")

    def get_dataframes(self):
        """
        Returns the individual training and testing DataFrames.
        """
        return self.train_data, self.test_data

    def get_concatenated_dataframe(self) -> pd.DataFrame:
        """
        Concatenates train and test dataframes into a single dataset,
        adding a column to track the original dataset split, and normalizes
        macronutrient totals for direct analysis.
        """
        if self.concatenated_dataset is not None:
            return self.concatenated_dataset

        train_df = self.train_data.copy()
        test_df = self.test_data.copy()

        train_df["dataset_split"] = "train"
        test_df["dataset_split"] = "test"

        self.concatenated_dataset = pd.concat([train_df, test_df], ignore_index=True)

        # Expand total_macronutrients dictionary into distinct columns for easy plotting
        macros_df = pd.json_normalize(self.concatenated_dataset["total_macronutrients"])
        self.concatenated_dataset = pd.concat([self.concatenated_dataset, macros_df], axis=1)

        expected_len = len(train_df) + len(test_df)
        if len(self.concatenated_dataset) != expected_len:
            print("WARNING: Data mismatch during concatenation.")

        return self.concatenated_dataset

    def get_exploded_ingredients(self) -> pd.DataFrame:
        """
        Explodes the nested list of ingredients per dish and extracts
        individual fields like name and weight_grams.
        """
        if self.exploded_ingredients is not None:
            return self.exploded_ingredients

        if self.concatenated_dataset is None:
            self.get_concatenated_dataframe()

        # Explode ingredients column
        exploded = self.concatenated_dataset.explode("ingredients").dropna(subset=["ingredients"])

        # Extract name and weight from the ingredient dicts
        exploded["ing_name"] = exploded["ingredients"].apply(
            lambda x: x.get("name", "").strip().lower() if isinstance(x, dict) else ""
        )
        exploded["ing_weight"] = exploded["ingredients"].apply(
            lambda x: float(x.get("weight_grams", 0.0)) if isinstance(x, dict) else 0.0
        )

        # Filter out records with empty ingredient names
        exploded = exploded[exploded["ing_name"] != ""]
        self.exploded_ingredients = exploded
        return self.exploded_ingredients

    def _plot_dataset_distribution(self):
        """
        Plots the global dataset splits (Train vs Test count of dishes),
        along with distributions of calories and total mass.
        """
        if self.concatenated_dataset is None:
            self.get_concatenated_dataframe()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.canvas.manager.set_window_title("Nutrition5k - Split & Nutrient Distributions")
        fig.suptitle("Nutrition5k Split and Nutrient Distributions", fontsize=16, fontweight="bold")

        # 1. Bar plot of split sizes
        split_counts = self.concatenated_dataset["dataset_split"].value_counts()
        colors = ["#4F46E5", "#EC4899"]  # Indigo & Rose
        split_counts.plot(kind="bar", color=colors, edgecolor="black", ax=axes[0], width=0.5)
        axes[0].set_title("Number of Dishes per Split", fontweight="bold")
        axes[0].set_ylabel("Dishes count")
        axes[0].set_xlabel("Dataset Split")
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
        axes[0].grid(axis="y", linestyle="--", alpha=0.3)
        for i, val in enumerate(split_counts):
            axes[0].text(i, val + (max(split_counts) * 0.01), f"{val}", ha="center", fontweight="bold")

        # 2. Histogram of total calories per split
        train_df = self.concatenated_dataset[self.concatenated_dataset["dataset_split"] == "train"]
        test_df = self.concatenated_dataset[self.concatenated_dataset["dataset_split"] == "test"]

        axes[1].hist(train_df["calories"], bins=30, alpha=0.6, label="Train", color="#4F46E5", edgecolor="none")
        axes[1].hist(test_df["calories"], bins=30, alpha=0.6, label="Test", color="#EC4899", edgecolor="none")
        axes[1].set_title("Total Calories Distribution", fontweight="bold")
        axes[1].set_xlabel("Calories (kcal)")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].grid(linestyle="--", alpha=0.3)

        # 3. Histogram of total mass per split
        axes[2].hist(train_df["total_mass_grams"], bins=30, alpha=0.6, label="Train", color="#4F46E5", edgecolor="none")
        axes[2].hist(test_df["total_mass_grams"], bins=30, alpha=0.6, label="Test", color="#EC4899", edgecolor="none")
        axes[2].set_title("Total Dish Mass Distribution", fontweight="bold")
        axes[2].set_xlabel("Mass (grams)")
        axes[2].set_ylabel("Frequency")
        axes[2].legend()
        axes[2].grid(linestyle="--", alpha=0.3)

        plt.tight_layout()

    def _plot_top_ingredients_by_mass(self):
        """
        Calculates and plots the top 30 ingredients by total mass (weight in kg),
        breaking down the contribution of train and test splits in a stacked bar chart.
        """
        exploded = self.get_exploded_ingredients()

        # Group by name and split, then sum weights
        mass_split = exploded.groupby(["ing_name", "dataset_split"])["ing_weight"].sum().unstack(fill_value=0.0)
        mass_split = mass_split / 1000.0  # Convert to kilograms
        mass_split["total_mass"] = mass_split["train"] + mass_split["test"]

        # Sort and take top 30
        top_30 = mass_split.sort_values(by="total_mass", ascending=True).tail(30)

        plt.figure("Nutrition5k - Top 30 Ingredients by Mass", figsize=(12, 10))
        ax = plt.gca()

        # Plot stacked bar chart with custom premium colors (Vibrant Blue & Cool Mint)
        top_30[["train", "test"]].plot(
            kind="barh",
            stacked=True,
            color=["#2563EB", "#10B981"],
            edgecolor="#1E293B",
            ax=ax,
            width=0.7
        )

        plt.title("Top 30 Ingredients by Total Mass in Nutrition5k", fontsize=15, fontweight="bold", pad=15)
        plt.xlabel("Total Mass (kg)", fontsize=12, fontweight="semibold")
        plt.ylabel("Ingredient", fontsize=12, fontweight="semibold")
        plt.legend(["Train Split", "Test Split"], loc="lower right", fontsize=10)
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()

    def _plot_top_ingredients_by_occurrence(self):
        """
        Calculates and plots the top 30 most frequent ingredients by occurrence count
        in dishes, breaking down train and test splits in a stacked bar chart.
        """
        exploded = self.get_exploded_ingredients()

        # Group by name and split, then count unique dishes
        occ_split = exploded.groupby(["ing_name", "dataset_split"])["food_id"].nunique().unstack(fill_value=0)
        occ_split["total_occurrences"] = occ_split["train"] + occ_split["test"]

        # Sort and take top 30
        top_30 = occ_split.sort_values(by="total_occurrences", ascending=True).tail(30)

        plt.figure("Nutrition5k - Top 30 Ingredients by Occurrence", figsize=(12, 10))
        ax = plt.gca()

        # Plot stacked bar chart with custom colors (Purple & Amber)
        top_30[["train", "test"]].plot(
            kind="barh",
            stacked=True,
            color=["#7C3AED", "#F59E0B"],
            edgecolor="#1E293B",
            ax=ax,
            width=0.7
        )

        plt.title("Top 30 Ingredients by Frequency of Occurrence", fontsize=15, fontweight="bold", pad=15)
        plt.xlabel("Number of Dishes (Count)", fontsize=12, fontweight="semibold")
        plt.ylabel("Ingredient", fontsize=12, fontweight="semibold")
        plt.legend(["Train Split", "Test Split"], loc="lower right", fontsize=10)
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()

    def _plot_ingredients_per_dish(self):
        """
        Analyzes and plots the dish complexity based on the number of ingredients.
        Uses horizontal boxplots to contrast Train and Test sets.
        """
        if self.concatenated_dataset is None:
            self.get_concatenated_dataframe()

        dish_complexity = self.concatenated_dataset.copy()
        dish_complexity["num_ingredients"] = dish_complexity["ingredients"].apply(len)

        plt.figure("Nutrition5k - Ingredients Complexity Per Dish", figsize=(10, 6))
        ax = plt.gca()

        train_complexity = dish_complexity[dish_complexity["dataset_split"] == "train"]["num_ingredients"]
        test_complexity = dish_complexity[dish_complexity["dataset_split"] == "test"]["num_ingredients"]

        # Boxplot styling
        box_style = dict(facecolor='#C7D2FE', color='#4F46E5', linewidth=1.5, alpha=0.8)
        median_style = dict(color='#EF4444', linewidth=2.5)
        whisker_style = dict(color='#4F46E5', linewidth=1.5, linestyle='--')
        cap_style = dict(color='#4F46E5', linewidth=1.5)
        flier_style = dict(marker='o', markerfacecolor='#EF4444', markersize=5, alpha=0.4, markeredgecolor='none')

        ax.boxplot(
            [train_complexity, test_complexity],
            vert=False,
            patch_artist=True,
            labels=["Train Split", "Test Split"],
            boxprops=box_style,
            medianprops=median_style,
            whiskerprops=whisker_style,
            capprops=cap_style,
            flierprops=flier_style
        )

        plt.title("Distribution of Number of Ingredients per Dish", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Number of Ingredients", fontsize=12, fontweight="semibold")
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()

    def _analyze_test_set_challenges(self):
        """
        Analyzes the challenges the model faces on the Nutrition5k Test Set:
        1. Out-of-Vocabulary (OOV) ingredients.
        2. Percentage of unseen recipe combinations.
        """
        if self.concatenated_dataset is None:
            self.get_concatenated_dataframe()

        # Compute unique sorted list of ingredient names per dish
        def get_recipe_signature(ing_list):
            if isinstance(ing_list, list):
                return tuple(sorted([
                    str(ing.get("name", "")).lower().strip()
                    for ing in ing_list if isinstance(ing, dict) and ing.get("name", "")
                ]))
            return ()

        df = self.concatenated_dataset.copy()
        df["recipe_signature"] = df["ingredients"].apply(get_recipe_signature)
        df["exploded_ingredients"] = df["recipe_signature"].apply(list)

        train_df = df[df["dataset_split"] == "train"]
        test_df = df[df["dataset_split"] == "test"].copy()

        # --- OUT-OF-VOCABULARY ANALYSIS ---
        train_vocab = set(train_df["exploded_ingredients"].explode().dropna())
        test_vocab = set(test_df["exploded_ingredients"].explode().dropna())

        oov_ingredients = test_vocab - train_vocab

        if oov_ingredients:
            test_exploded = test_df.explode("exploded_ingredients")
            oov_counts = test_exploded[test_exploded["exploded_ingredients"].isin(oov_ingredients)][
                "exploded_ingredients"].value_counts()

            plt.figure("Nutrition5k - OOV Ingredients in Test Set", figsize=(10, 6))
            oov_counts.head(20)[::-1].plot(kind="barh", color="#EF4444", edgecolor="black", width=0.7)
            plt.title("Top 20 Out-Of-Vocabulary (OOV) Ingredients in Test Set", fontsize=14, fontweight="bold")
            plt.xlabel("Occurrences in Test Set", fontsize=12)
            plt.ylabel("Ingredient Name", fontsize=12)
            plt.grid(axis="x", linestyle="--", alpha=0.4)
            plt.tight_layout()
            print(f"[Analysis] Found {len(oov_ingredients)} OOV ingredients in the test set.")
        else:
            print("[Analysis] No Out-Of-Vocabulary (OOV) ingredients found in the test set!")

        # --- NOVEL COMBINATIONS (RECIPE SIGNATURES) ---
        train_signatures = set(train_df["recipe_signature"])
        test_signatures = test_df["recipe_signature"]

        novel_sigs = [sig for sig in test_signatures if sig not in train_signatures]
        novel_pct = (len(novel_sigs) / len(test_signatures)) * 100 if len(test_signatures) > 0 else 0.0

        print(f"[Analysis] Percentage of unseen recipe ingredient combinations in Test Set: {novel_pct:.2f}%")

        # Plot a visual metric panel for novelty percentage
        plt.figure("Nutrition5k - Test Set Novelty Metric", figsize=(6, 4))
        plt.text(0.5, 0.6, f"{novel_pct:.1f}%", fontsize=48, fontweight="bold", ha="center", va="center", color="#3B82F6")
        plt.text(
            0.5, 0.25,
            "Of Test Dishes consist of recipe combinations\ncompletely unseen in the Training Set.",
            fontsize=10, ha="center", va="center", color="#1F2937"
        )
        plt.axis("off")
        plt.tight_layout()

    def plot_dataset_analysis(self):
        """
        Runs the full pipeline of plots and shows them on screen.
        """
        self._plot_dataset_distribution()
        self._plot_top_ingredients_by_mass()
        self._plot_top_ingredients_by_occurrence()
        self._plot_ingredients_per_dish()
        self._plot_ingredient_cooccurrence_network()
        self._analyze_test_set_challenges()

        plt.show()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Relative paths to run it from either datasets/Nutrition5k or root directory
    TRAIN_PATH = "data/ground truth/train_dataset.json"
    TEST_PATH = "data/ground truth/test_dataset.json"

    if not os.path.exists(TRAIN_PATH):
        TRAIN_PATH = os.path.join(base_dir, "data/ground truth/train_dataset.json")
    if not os.path.exists(TEST_PATH):
        TEST_PATH = os.path.join(base_dir, "data/ground truth/test_dataset.json")

    print(f"Loading files:\n- Train: {TRAIN_PATH}\n- Test: {TEST_PATH}")

    try:
        Nutrition5k_analysis = AnalyzeNutrition5k(TRAIN_PATH, TEST_PATH)
        Nutrition5k_analysis.plot_dataset_analysis()
    except FileNotFoundError as e:
        print(f"Error: {e}.\nPlease ensure train_dataset.json and test_dataset.json exist.")
