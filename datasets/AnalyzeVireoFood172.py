import os
import pandas as pd
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


class AnalyzeDataset:
    """
    A class used to load, analyze, and visualize distributions and complexities
    of a food dataset containing images and ingredients.
    """

    def __init__(self, train_path: str, val_path: str, test_path: str):
        """
        Initializes the AnalyzeDataset instance by loading JSON data files.
        """
        self.train_data = self._load_json(train_path, "training")
        self.val_data = self._load_json(val_path, "validation")
        self.test_data = self._load_json(test_path, "testing")

        self.concatenated_dataset = None
        self.dataset_distribution = None
        self.ingredients_distribution = None

    def _load_json(self, path: str, dataset_name: str) -> pd.DataFrame:
        """
        Private helper to validate and load a JSON file into a DataFrame.
        """
        if path is not None and os.path.exists(path):
            return pd.read_json(path)
        raise FileNotFoundError(f"Il dataset JSON di {dataset_name} non esiste.")

    def get_dataframes(self):
        """
        Returns the individual training, validation, and testing DataFrames.
        """
        return self.train_data, self.val_data, self.test_data

    def get_concatenated_dataframe(self) -> pd.DataFrame:
        """
        Concatenates train, val, and test dataframes into a single dataset,
        adding a column to track the original dataset split.
        """
        train_df = self.train_data.copy()
        val_df = self.val_data.copy()
        test_df = self.test_data.copy()

        train_df["dataset_split"] = "train"
        val_df["dataset_split"] = "val"
        test_df["dataset_split"] = "test"

        self.concatenated_dataset = pd.concat([train_df, val_df, test_df], ignore_index=True)

        expected_len = len(train_df) + len(val_df) + len(test_df)
        if len(self.concatenated_dataset) != expected_len:
            print("ATTENZIONE! Alcuni dati sono andati persi.")

        return self.concatenated_dataset

    def _plot_horizontal_bar(self, data: pd.DataFrame, window_title: str, plot_title: str,
                             xlabel: str, ylabel: str, figsize=(12, 8), stacked=True):
        """
        Private helper to generate and configure horizontal bar plots.
        """
        plt.figure(window_title, figsize=figsize)
        data.plot(kind="barh", stacked=stacked, ax=plt.gca())

        plt.title(plot_title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()

    def _plot_horizontal_boxplot(self, data: pd.DataFrame, window_title: str, plot_title: str,
                                 xlabel: str, ylabel: str, figsize=(12, 8)):
        """
        Private helper to generate and style horizontal boxplots.
        """
        box_style = dict(facecolor='#85C1E9', color='#2874A6', linewidth=1.5, alpha=0.8)
        median_style = dict(color='#C0392B', linewidth=2.5)
        whisker_style = dict(color='#2874A6', linewidth=1.5, linestyle='--')
        cap_style = dict(color='#2874A6', linewidth=1.5)
        flier_style = dict(marker='o', markerfacecolor='#E74C3C', markersize=5, alpha=0.5, markeredgecolor='none')

        plt.figure(window_title, figsize=figsize)
        data.boxplot(
            column="num_ingredients",
            by="food_category",
            vert=False,
            ax=plt.gca(),
            patch_artist=True,
            boxprops=box_style,
            medianprops=median_style,
            whiskerprops=whisker_style,
            capprops=cap_style,
            flierprops=flier_style
        )
        plt.title(plot_title)
        plt.suptitle("")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()

    def _plot_dataset_distribution(self):
        """
        Plots the global, top 20, and bottom 20 distributions of images
        categorized by food category and dataset split.
        """
        if self.concatenated_dataset is None:
            self.get_concatenated_dataframe()

        self.dataset_distribution = pd.crosstab(
            index=self.concatenated_dataset["food_category"],
            columns=self.concatenated_dataset["dataset_split"]
        )

        self.dataset_distribution = self.dataset_distribution[['train', 'val', 'test']]
        totals = self.dataset_distribution.sum(axis=1)
        self.dataset_distribution = self.dataset_distribution.loc[totals.sort_values(ascending=False).index]

        plt.figure("Distribuzione Globale Immagini", figsize=(15, 6))
        self.dataset_distribution.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.title("Distribuzione delle Immagini per Categoria di Cibo e Split")
        plt.xlabel("Categoria di Cibo")
        plt.ylabel("Numero di Immagini")
        plt.xticks([])
        plt.tight_layout()

        self._plot_horizontal_bar(
            data=self.dataset_distribution.head(20)[::-1],
            window_title="Top 20 Cibi",
            plot_title="Top 20 Categorie di Cibo per Numero di Immagini",
            xlabel="Numero di Immagini",
            ylabel="Categoria di Cibo"
        )

        self._plot_horizontal_bar(
            data=self.dataset_distribution.tail(20)[::-1],
            window_title="Bottom 20 Cibi",
            plot_title="Bottom 20 Categorie di Cibo per Numero di Immagini",
            xlabel="Numero di Immagini",
            ylabel="Categoria di Cibo"
        )

    def _plot_ingredients_distribution(self):
        """
        Extracts individual ingredients, calculates their frequency, and plots
        the global, top 20, and bottom 20 occurrences.
        """
        if self.concatenated_dataset is None:
            self.get_concatenated_dataframe()

        exploded_df = self.concatenated_dataset.explode("ingredients").dropna(subset=["ingredients"])
        exploded_df["ingredients"] = exploded_df["ingredients"].str.lower().str.strip()
        exploded_df = exploded_df.reset_index(drop=True)

        self.ingredients_distribution = pd.crosstab(
            index=exploded_df["ingredients"],
            columns=exploded_df["dataset_split"]
        )

        self.ingredients_distribution = self.ingredients_distribution[['train', 'val', 'test']]
        totals = self.ingredients_distribution.sum(axis=1)
        self.ingredients_distribution = self.ingredients_distribution.loc[totals.sort_values(ascending=False).index]

        plt.figure("Distribuzione Globale Ingredienti", figsize=(15, 6))
        self.ingredients_distribution.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.title("Distribuzione Globale di Tutti gli Ingredienti")
        plt.xlabel("Ingrediente")
        plt.ylabel("Numero di Occorrenze")
        plt.xticks([])
        plt.semilogy()
        plt.tight_layout()

        self._plot_horizontal_bar(
            data=self.ingredients_distribution.head(20)[::-1],
            window_title="Top 20 Ingredienti",
            plot_title="Top 20 Ingredienti per Occorrenza",
            xlabel="Numero di Occorrenze",
            ylabel="Ingrediente"
        )

        self._plot_horizontal_bar(
            data=self.ingredients_distribution.tail(20)[::-1],
            window_title="Bottom 20 Ingredienti",
            plot_title="Bottom 20 Ingredienti per Occorrenza",
            xlabel="Numero di Occorrenze",
            ylabel="Ingrediente"
        )

    def _plot_ingredients_distribution_per_dish(self):
        """
        Analyzes and plots the complexity of dishes based on the number of ingredients,
        utilizing global bar charts and top/bottom 20 boxplots.
        """
        if self.concatenated_dataset is None:
            self.get_concatenated_dataframe()

        food_complexity = self.concatenated_dataset.copy()
        food_complexity["num_ingredients"] = food_complexity["ingredients"].str.len()

        complexity_distribution = pd.crosstab(
            index=food_complexity["num_ingredients"],
            columns=food_complexity["dataset_split"]
        )
        complexity_distribution = complexity_distribution[['train', 'val', 'test']]

        plt.figure("Ingredienti Globali Per Piatto", figsize=(15, 6))
        complexity_distribution.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.title("Distribuzione Globale degli Ingredienti per Piatto")
        plt.xlabel("Numero di Ingredienti")
        plt.ylabel("Numero di Piatti")
        plt.semilogy()
        plt.tight_layout()

        median_complexity = food_complexity.groupby("food_category")["num_ingredients"].median().sort_values(
            ascending=False)

        self._plot_horizontal_boxplot(
            data=food_complexity[food_complexity["food_category"].isin(median_complexity.head(20).index)],
            window_title="Top 20 Piatti Complessi",
            plot_title="Top 20 Categorie per Complessità del Piatto",
            xlabel="Numero di Ingredienti",
            ylabel="Categoria di Cibo"
        )

        self._plot_horizontal_boxplot(
            data=food_complexity[food_complexity["food_category"].isin(median_complexity.tail(20).index)],
            window_title="Bottom 20 Piatti Semplici",
            plot_title="Bottom 20 Categorie per Complessità del Piatto",
            xlabel="Numero di Ingredienti",
            ylabel="Categoria di Cibo"
        )

    def _plot_ingredient_cooccurrence_network(self, top_n=50):
        """
        Builds and plots a co-occurrence network graph displaying the most
        frequent ingredient pairings in the dataset.
        """
        if self.concatenated_dataset is None:
            self.get_concatenated_dataframe()

        ingredients_network = self.concatenated_dataset.dropna(subset=["ingredients"]).copy()

        def clean_list(lst):
            if isinstance(lst, list):
                return [str(i).lower().strip() for i in lst]
            return []

        ingredients_network["clean_ingredients"] = ingredients_network["ingredients"].apply(clean_list)
        pair_counter = Counter()

        for ing_list in ingredients_network["clean_ingredients"]:
            sorted_ings = sorted(ing_list)
            pairs = itertools.combinations(sorted_ings, 2)
            pair_counter.update(pairs)

        G = nx.Graph()
        for (ing1, ing2), weight in pair_counter.most_common(top_n):
            G.add_edge(ing1, ing2, weight=weight)

        plt.figure(f"Rete Ingredienti (Top {top_n})", figsize=(16, 12))
        pos = nx.kamada_kawai_layout(G)

        weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(weights) if weights else 1
        edge_widths = [(w / max_weight) * 5 + 1 for w in weights]

        nx.draw_networkx_nodes(
            G, pos,
            node_size=400,
            node_color="#85C1E9",
            edgecolors="#2874A6",
            linewidths=1.5,
            alpha=0.9
        )

        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            edge_color="#7F8C8D",
            alpha=0.6
        )

        pos_labels = {nodo: (coordinate[0], coordinate[1] + 0.04) for nodo, coordinate in pos.items()}
        nx.draw_networkx_labels(
            G, pos_labels,
            font_size=8,
            font_weight="bold",
            font_family="sans-serif",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5)
        )

        plt.title(f"Rete di Co-occorrenza degli Ingredienti (Top {top_n} Coppie)", fontsize=16)
        plt.axis("off")
        plt.tight_layout()

    def plot_dataset_analysis(self):
        self._plot_dataset_distribution()
        self._plot_ingredients_distribution()
        self._plot_ingredients_distribution_per_dish()
        self._plot_ingredient_cooccurrence_network()

        plt.show()


if __name__ == "__main__":
    TRAIN_PATH = "VireoFood172/data/ground truth/train_dataset.json"
    VAL_PATH = "VireoFood172/data/ground truth/val_dataset.json"
    TEST_PATH = "VireoFood172/data/ground truth/test_dataset.json"

    VireoFood = AnalyzeDataset(TRAIN_PATH, VAL_PATH, TEST_PATH)
    VireoFood.plot_dataset_analysis()