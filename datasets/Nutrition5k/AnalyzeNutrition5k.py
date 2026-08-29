import os
import sys
import pandas as pd
import numpy as np
import matplotlib

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class AnalyzeNutrition5k:
    """
    Classe per l'analisi esplorativa (EDA) del dataset Nutrition5k, 
    unendo i set di training e testing per fornire statistiche complete 
    sulla complessità dei piatti, sui valori nutrizionali e sugli ingredienti.
    """

    def __init__(self, train_path: str, test_path: str):
        self.train_data = self._load_json(train_path, "training")
        self.test_data = self._load_json(test_path, "testing")

        # Unione dei dataset come richiesto (unendo risultati test_dataset e train_dataset)
        self.dataset = pd.concat([self.train_data, self.test_data], ignore_index=True)

        # Estrazione dei macronutrienti per un accesso più semplice
        macros_df = pd.json_normalize(self.dataset["total_macronutrients"])
        self.dataset = pd.concat([self.dataset, macros_df], axis=1)

    def _load_json(self, path: str, dataset_name: str) -> pd.DataFrame:
        if path is not None and os.path.exists(path):
            return pd.read_json(path)
        raise FileNotFoundError(f"Dataset {dataset_name} non trovato al percorso: {path}")

    def plot_dish_complexity(self, save_dir: str = None):
        """
        1. Complessità dei piatti (Numero di ingredienti)
        Istogramma a barre che mostra quanti ingredienti ci sono in media in un piatto.
        Dimostra che il task non è banale.
        """
        # Calcolo del numero di ingredienti validi per ogni piatto
        num_ingredients = self.dataset["ingredients"].apply(
            lambda x: len([ing for ing in x if isinstance(ing, dict) and ing.get("name", "").strip()]) if isinstance(x,
                                                                                                                     list) else 0
        )

        # Suddivisione in categorie: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12+
        bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, np.inf]
        labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15+']
        complexity_cats = pd.cut(num_ingredients, bins=bins, labels=labels, right=True)

        counts = complexity_cats.value_counts().reindex(labels).fillna(0)

        plt.figure("Complessità dei Piatti", figsize=(8, 6))
        bars = plt.bar(counts.index, counts.values, color='#4F46E5', edgecolor='black', alpha=0.85)

        plt.title("Complessità dei Piatti (Numero di Ingredienti)", fontsize=14, fontweight='bold')
        plt.xlabel("Numero di Ingredienti nel Piatto", fontsize=12)
        plt.ylabel("Numero di Immagini", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        # Aggiunta dei valori esatti sopra ogni barra
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + (max(counts.values) * 0.02),
                     int(yval), ha='center', va='bottom', fontweight='bold', color='#1E293B')

        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, "1_complessita_piatti.png"), dpi=300)

    def plot_nutritional_distribution(self, save_dir: str = None):
        """
        2. Distribuzione dei Valori Nutrizionali
        Grafico che mostra la distribuzione di Calorie (Kcal) e Massa totale (grammi).
        Aiuta a capire se il dataset è sbilanciato.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.canvas.manager.set_window_title("Distribuzione Valori Nutrizionali")

        # Calorie
        axes[0].hist(self.dataset["calories"], bins=40, color="#EC4899", edgecolor="black", alpha=0.7)
        axes[0].set_title("Distribuzione delle Calorie", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("Calorie (Kcal)", fontsize=11)
        axes[0].set_ylabel("Frequenza (Numero di Piatti)", fontsize=11)
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        # Massa Totale
        axes[1].hist(self.dataset["total_mass_grams"], bins=40, color="#10B981", edgecolor="black", alpha=0.7)
        axes[1].set_title("Distribuzione della Massa Totale", fontsize=13, fontweight='bold')
        axes[1].set_xlabel("Massa (grammi)", fontsize=11)
        axes[1].set_ylabel("Frequenza (Numero di Piatti)", fontsize=11)
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, "2_distribuzione_nutrizionale.png"), dpi=300)

    def plot_top_ingredients(self, top_n: int, save_dir: str = None):
        """
        3. Gli Ingredienti più frequenti
        Grafico a barre orizzontali con gli ingredienti più presenti nel dataset.
        Evidenzia il "bias" culturale del dataset (es. dieta californiana).
        """
        exploded = self.dataset.explode("ingredients").dropna(subset=["ingredients"])

        ingredient_names = exploded["ingredients"].apply(
            lambda x: x.get("name", "").strip().lower() if isinstance(x, dict) else ""
        )

        ingredient_names = ingredient_names[ingredient_names != ""]
        ingredient_counts = ingredient_names.value_counts().head(top_n)
        ingredient_counts = ingredient_counts.sort_values(ascending=True)

        plt.figure(f"Top {top_n} Ingredienti", figsize=(10, 8))
        ingredient_counts.plot(kind='barh', color='#4F46E5', edgecolor='black', alpha=0.85)

        plt.title(f"I {top_n} Ingredienti più Frequenti nel Dataset", fontsize=14, fontweight='bold')
        plt.xlabel("Numero di Occorrenze", fontsize=12)
        plt.ylabel("Ingrediente", fontsize=12)
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"3_top_{top_n}_ingredienti.png"), dpi=300)

    def analyze_invisible_ingredients(self, save_dir: str = None):
        """
        4. Presenza di ingredienti "invisibili"
        Stampa a console alcune statistiche su ingredienti nascosti per giustificare
        l'uso di modelli avanzati in grado di dedurli. Inoltre salva un grafico a barre.
        """

        exploded = self.dataset.explode("ingredients").dropna(subset=["ingredients"])
        ingredient_names = exploded["ingredients"].apply(
            lambda x: x.get("name", "").strip().lower() if isinstance(x, dict) else ""
        )
        ingredient_names = ingredient_names[ingredient_names != ""]

        # Esempi di keyword per ingredienti invisibili espansi (come erano nel file prima)
        invisible_keywords = [
            "agave nectar", "apple juice", "balsamic vinegar", "barbecue sauce",
            "broth", "brown sugar", "butter", "caesar dressing", "cocoa",
            "corn starch", "cranberry juice", "fish oil", "flour", "french dressing",
            "grape juice", "grapefruit juice", "gravy", "honey", "italian dressing",
            "lemon juice", "maple syrup", "margarine", "nectar", "oil", "olive oil",
            "orange juice", "pepper", "protein powder", "ranch dressing",
            "salad dressing", "salt", "soy sauce", "sugar", "syrup", "teriyaki sauce",
            "vegetable juice", "vegetable oil", "vinaigrette", "vinegar", "water"
        ]

        counts = {}
        print("\nFrequenza di alcuni ingredienti 'invisibili' (o che li contengono) nel dataset:")
        for kw in invisible_keywords:
            count = ingredient_names.str.contains(kw).sum()
            if count > 0:  # Evitiamo di graficare quelli a zero per chiarezza
                counts[kw.capitalize()] = count
                print(f" - {kw.capitalize()}: {count} occorrenze")
        
        print("="*70 + "\n")

        # Generazione del grafico (prendiamo solo i top 20 per non affollare troppo il grafico)
        plt.figure("Ingredienti Invisibili", figsize=(10, 8))
        
        # Ordiniamo e prendiamo i top 20
        sorted_counts_list = sorted(counts.items(), key=lambda item: item[1], reverse=False)
        top_invisible = dict(sorted_counts_list[-20:])
        
        plt.barh(list(top_invisible.keys()), list(top_invisible.values()), color='#4F46E5', edgecolor='black', alpha=0.85)
        plt.title("Top 20 Ingredienti 'Invisibili' o Impliciti", fontsize=14, fontweight='bold')
        plt.xlabel("Numero di Occorrenze", fontsize=12)
        plt.ylabel("Ingrediente", fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, "4_ingredienti_invisibili.png"), dpi=300)

    def run_all_analyses(self, top_n: int, save_dir: str = None, show_plots: bool = True):
        # 1. Complessità
        self.plot_dish_complexity(save_dir)
        # 2. Valori nutrizionali
        self.plot_nutritional_distribution(save_dir)
        # 3. Ingredienti frequenti
        self.plot_top_ingredients(top_n=top_n, save_dir=save_dir)
        # 4. Ingredienti invisibili
        self.analyze_invisible_ingredients(save_dir=save_dir)
        
        # Mostra tutti i grafici se richiesto
        if show_plots:
            plt.show()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    TRAIN_PATH = os.path.join(base_dir, "data", "ground truth", "train_dataset.json")
    TEST_PATH = os.path.join(base_dir, "data", "ground truth", "test_dataset.json")

    # Directory di output per i grafici
    OUTPUT_DIR = os.path.join(base_dir, "eda_plots")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Caricamento file in corso...\n- Train: {TRAIN_PATH}\n- Test: {TEST_PATH}")
    print(f"I grafici verranno salvati in: {OUTPUT_DIR}")

    try:
        # =====================================================================
        # SPECIFICARE QUI QUANTI INGREDIENTI MOSTRARE NEL GRAFICO "TOP N" 
        # (es. 10, 20 o 30)
        TOP_N = 20
        # =====================================================================

        analyzer = AnalyzeNutrition5k(TRAIN_PATH, TEST_PATH)

        # Esegui l'analisi (se passiamo --save-only non mostreremo l'interfaccia UI)
        show_ui = "--save-only" not in sys.argv
        analyzer.run_all_analyses(top_n=TOP_N, save_dir=OUTPUT_DIR, show_plots=show_ui)

        print(f"\nAnalisi completata con successo! Grafici salvati in {OUTPUT_DIR}")

    except FileNotFoundError as e:
        print(f"Errore: {e}")
