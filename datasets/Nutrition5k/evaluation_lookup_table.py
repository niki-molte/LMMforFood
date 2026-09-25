import argparse
import json
import os
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer, util

class EvaluationLookupTable:
    def __init__(
        self, 
        metadata_csv_path: str, 
        model_name: str = 'all-MiniLM-L6-v2', 
        similarity_threshold: float = 0.6
    ):
        """
        Inizializza l'evaluator per il Deterministic RAG.
        
        Args:
            metadata_csv_path: Percorso al file ingredients_metadata.csv
            model_name: Modello di sentence-transformers da usare per gli embedding
            similarity_threshold: Soglia minima di similarità coseno per accettare un ingrediente
        """
        self.similarity_threshold = similarity_threshold
        
        # 1. Caricamento del file CSV di metadata
        print(f"Caricamento vocabolario ingredienti da: {metadata_csv_path}")
        self.metadata_df = pd.read_csv(metadata_csv_path)
        
        # Verifica delle colonne attese
        expected_cols = ['ingr', 'id', 'cal/g', 'fat(g)', 'carb(g)', 'protein(g)']
        for col in expected_cols:
            if col not in self.metadata_df.columns:
                raise ValueError(f"Colonna mancante nel CSV: {col}")
                
        # Rimuoviamo eventuali duplicati (stesso nome di ingrediente) tenendo il primo
        self.metadata_df = self.metadata_df.drop_duplicates(subset=['ingr'], keep='first')
        
        self.ingredients_vocab = self.metadata_df['ingr'].tolist()
        
        # Dizionario di lookup per accesso rapido in O(1) ai macronutrienti per grammo
        self.lookup_dict = self.metadata_df.set_index('ingr').to_dict('index')
        
        # 2. Setup del modello di Semantic Matching (RAG)
        print(f"Inizializzazione modello SentenceTransformer: {model_name}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Utilizzo device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        
        print("Generazione embedding per il vocabolario degli ingredienti...")
        self.vocab_embeddings = self.model.encode(self.ingredients_vocab, convert_to_tensor=True)
        print("Inizializzazione completata.\n")

    def find_best_match(self, predicted_ingredient: str) -> Tuple[Optional[str], float]:
        """
        Calcola la similarità coseno tra l'ingrediente predetto e il vocabolario del CSV.
        
        Returns:
            Tuple con (nome_ingrediente_matchato, punteggio_similarita).
            Se il punteggio < threshold, il nome_ingrediente_matchato sarà None.
        """
        if not predicted_ingredient or not isinstance(predicted_ingredient, str):
            return None, 0.0
            
        pred_embedding = self.model.encode(predicted_ingredient, convert_to_tensor=True)
        
        # Calcola la similarità coseno con tutti gli embedding del vocabolario
        cosine_scores = util.cos_sim(pred_embedding, self.vocab_embeddings)[0]
        
        # Trova il miglior match
        best_idx = torch.argmax(cosine_scores).item()
        best_score = cosine_scores[best_idx].item()
        best_match_ingr = self.ingredients_vocab[best_idx]
        
        if best_score >= self.similarity_threshold:
            return best_match_ingr, best_score
        else:
            return None, best_score

    def extract_predicted_ingredients(self, dish_pred: Dict) -> List[Dict]:
        """
        Estrae la lista degli ingredienti predetti da un dizionario JSON.
        Gestisce dinamicamente diverse possibili strutture del JSON.
        """
        items = []

        # Pattern 1a: Struttura Two-Step/Hybrid con "predicted_nutrition_step2"
        # (ha precedenza su predicted_nutrition perché è il risultato finale del pipeline)
        if 'predicted_nutrition_step2' in dish_pred:
            nut_pred = dish_pred['predicted_nutrition_step2']
            if 'visible_ingredients' in nut_pred:
                items.extend(nut_pred['visible_ingredients'])
            if 'hidden_ingredients' in nut_pred:
                items.extend(nut_pred['hidden_ingredients'])

        # Pattern 1b: Struttura con "predicted_nutrition" (Standard/YOLO)
        elif 'predicted_nutrition' in dish_pred:
            nut_pred = dish_pred['predicted_nutrition']
            if 'visible_ingredients' in nut_pred:
                items.extend(nut_pred['visible_ingredients'])
            if 'hidden_ingredients' in nut_pred:
                items.extend(nut_pred['hidden_ingredients'])

        # Pattern 2: Liste piatte con chiavi generiche
        elif 'ingredients' in dish_pred:
            items.extend(dish_pred['ingredients'])
        elif 'predicted_ingredients' in dish_pred:
            items.extend(dish_pred['predicted_ingredients'])

        return items

    def process_dish(self, dish_pred: Dict) -> Dict:
        """
        Elabora la singola predizione di un piatto:
        - Estrae gli ingredienti e i pesi
        - Esegue la validazione tramite Semantic Matching (RAG)
        - Calcola deterministicamente i macronutrienti totali
        """
        ingredients_list = self.extract_predicted_ingredients(dish_pred)
        
        # Variabili per il Calcolo Deterministico
        total_cal = 0.0
        total_fat = 0.0
        total_carb = 0.0
        total_prot = 0.0
        
        validated_ingredients = []
        
        for item in ingredients_list:
            if not isinstance(item, dict):
                continue
                
            # Verifica chiavi per nome ingrediente ("ingredient" o "name") e peso ("weight_grams")
            pred_name = item.get('ingredient') or item.get('name')
            weight_grams = item.get('weight_grams')
            
            if not pred_name or weight_grams is None:
                continue
                
            try:
                weight_grams = float(weight_grams)
            except (ValueError, TypeError):
                continue
                
            # STEP 1: Semantic Matching (RAG)
            best_match_ingr, score = self.find_best_match(pred_name)
            
            ingr_result = {
                "original_prediction": pred_name,
                "weight_grams": weight_grams,
                "similarity_score": round(score, 4)
            }
            
            if best_match_ingr:
                # Match valido (>= 0.6): Recuperiamo i valori dal CSV
                macros_per_g = self.lookup_dict[best_match_ingr]
                
                # STEP 2: Calcolo Deterministico per l'ingrediente
                cal = weight_grams * macros_per_g['cal/g']
                fat = weight_grams * macros_per_g['fat(g)']
                carb = weight_grams * macros_per_g['carb(g)']
                prot = weight_grams * macros_per_g['protein(g)']
                
                # Aggiorniamo i totali del piatto
                total_cal += cal
                total_fat += fat
                total_carb += carb
                total_prot += prot
                
                ingr_result.update({
                    "matched_ingredient": best_match_ingr,
                    "status": "accepted",
                    "deterministic_macronutrients": {
                        "calories": round(cal, 4),
                        "fat": round(fat, 4),
                        "carbohydrates": round(carb, 4),
                        "protein": round(prot, 4)
                    }
                })
            else:
                # Sotto soglia: usiamo i macronutrienti originali predetti dal LLM come fallback
                orig_macros = item.get('macronutrients', {})
                if not isinstance(orig_macros, dict):
                    orig_macros = {}
                    
                try:
                    cal = float(orig_macros.get('calories', 0.0) or 0.0)
                    fat = float(orig_macros.get('fat', 0.0) or 0.0)
                    carb = float(orig_macros.get('carbohydrates', 0.0) or 0.0)
                    prot = float(orig_macros.get('protein', 0.0) or 0.0)
                except (ValueError, TypeError):
                    cal = fat = carb = prot = 0.0
                
                # Aggiorniamo i totali del piatto con i valori fallback originali
                total_cal += cal
                total_fat += fat
                total_carb += carb
                total_prot += prot
                
                ingr_result.update({
                    "matched_ingredient": None,
                    "status": "unmatched",
                    "deterministic_macronutrients": {
                        "calories": round(cal, 4),
                        "fat": round(fat, 4),
                        "carbohydrates": round(carb, 4),
                        "protein": round(prot, 4)
                    }
                })
                
            validated_ingredients.append(ingr_result)
            
        # Aggiungiamo i risultati post-processati al dizionario originale del piatto
        dish_pred['rag_validated_ingredients'] = validated_ingredients
        dish_pred['deterministic_total_macronutrients'] = {
            "calories": round(total_cal, 4),
            "fat": round(total_fat, 4),
            "carbohydrates": round(total_carb, 4),
            "protein": round(total_prot, 4)
        }
        
        return dish_pred

    def process_file(self, input_json_path: str, output_json_path: str) -> List[Dict]:
        """
        Legge un file JSON di predizioni, elabora ogni piatto e salva l'output aggiornato.
        """
        print(f"Elaborazione predizioni da: {input_json_path}")
        with open(input_json_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
            
        # Gestisce sia singoli oggetti JSON che liste di oggetti
        is_single_object = isinstance(predictions, dict)
        if is_single_object:
            predictions = [predictions]
            
        processed_data = []
        for i, dish_pred in enumerate(predictions):
            processed_dish = self.process_dish(dish_pred)
            processed_data.append(processed_dish)
            
        if is_single_object:
            processed_data = processed_data[0]
            
        # Salvataggio dell'output
        os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4)
            
        print(f"Elaborazione completata! File salvato in: {output_json_path}")
        return processed_data

    def process_directory(self, input_dir: str, output_dir: str):
        """
        Processa tutti i file JSON in una directory e li salva nella directory di output.
        """
        import glob
        os.makedirs(output_dir, exist_ok=True)
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        
        print(f"Trovati {len(json_files)} file in {input_dir}")
        for file_path in json_files:
            filename = os.path.basename(file_path)
            out_path = os.path.join(output_dir, filename)
            self.process_file(file_path, out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic RAG Post-Processing per predizioni LMM.")
    parser.add_argument("--metadata", type=str, required=True, 
                        help="Percorso al file ingredients_metadata.csv")
    parser.add_argument("--input", type=str, required=False, 
                        help="Percorso al singolo file JSON contenente le predizioni")
    parser.add_argument("--output", type=str, required=False, 
                        help="Percorso dove salvare il singolo file JSON processato")
    parser.add_argument("--batch_dir", type=str, required=False, 
                        help="Percorso alla cartella prediction (es. datasets/Nutrition5k/data/prediction) per elaborare tutte le sottocartelle in cartelle _rag")
    parser.add_argument("--threshold", type=float, default=0.6, 
                        help="Soglia di similarità coseno (default: 0.6)")
    
    args = parser.parse_args()
    
    evaluator = EvaluationLookupTable(
        metadata_csv_path=args.metadata,
        similarity_threshold=args.threshold
    )
    
    if args.batch_dir:
        print(f"\nInizio elaborazione BATCH per la cartella base: {args.batch_dir}")
        for item in os.listdir(args.batch_dir):
            input_dir = os.path.join(args.batch_dir, item)
            
            # Saltiamo i file singoli e le cartelle che sono già "_rag"
            if not os.path.isdir(input_dir) or item.endswith("_rag"):
                continue
                
            output_dir = os.path.join(args.batch_dir, f"{item}_rag")
            print(f"\n--- Elaborazione sottocartella: {item} -> {item}_rag ---")
            evaluator.process_directory(input_dir, output_dir)
            
        print("\nElaborazione BATCH completata con successo!")
        
    elif args.input and args.output:
        evaluator.process_file(args.input, args.output)
    else:
        print("Errore: Devi specificare o (--input e --output) oppure --batch_dir.")




#python evaluation_lookup_table.py --metadata data/metadata/ingredients_metadata.csv --batch_dir data/optuna/few_shot_standard