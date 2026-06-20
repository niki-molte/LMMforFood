import json
import os
import time
import unicodedata

from OllamaClient.gemma_client import GemmaClient
from OllamaClient.qwen_client import QwenClient
from OllamaClient.miniCPM_client import MiniCPMClient
from answer_schema.extraction import Macronutrients, IngredientWithNutrition, NutritionExtraction

PROVIDER = "minicpm"

# impostazioni checkpoint
SAVE_EVERY_N_ITEMS = 5
MAX_ITEMS_TO_TEST = 3

# percorsi
IMAGES_BASE_DIR = "/media/niki/Niki My Passport/scuola/università/magistrale/tesi/dataset/Nutrition5k/nutrition5k_dataset"
TEST_DATASET_PATH = "datasets/Nutrition5k/data/ground truth/test_dataset.json"
OUTPUT_PREDICTIONS_PATH = "data/prediction/test_predictions_nutrition5k_gemma4e2b_prova.json"


# ==========================================
# --- LOGICA DI ESECUZIONE ---
# ==========================================
def normalize_path(path: str) -> str:
    """
    Normalizza i caratteri Unicode nel percorso (NFC) per evitare
    mismatch di encoding tra il filesystem e le stringhe Python.
    """
    return unicodedata.normalize("NFC", path)


def save_json_safely(data, filepath):
    """Funzione atomica per salvare dati JSON in modo sicuro."""
    temp_filepath = filepath + ".tmp"
    try:
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(temp_filepath, filepath)
    except Exception as e:
        print(f"[ERRORE] Impossibile salvare il file JSON: {e}")


def process_single_image(item, client, system_instruction, few_shot_examples, processed_images):
    """Processa una singola immagine in modo sequenziale con logging di debug dettagliato."""
    img_path = item["image_path"]

    if img_path in processed_images:
        return None

    # Ricostruisce il path assoluto unendo la base dir al path relativo salvato nel JSON.
    # normalize_path evita mismatch Unicode (es. 'università' → 'universit?' su certi terminali).
    full_image_path = normalize_path(os.path.join(IMAGES_BASE_DIR, img_path))

    if not os.path.exists(full_image_path):
        print(f"[Avviso] File non trovato: {full_image_path}")
        return None

    prompt = (
        "Analyze this dish. Provide the full breakdown of visible/hidden ingredients, "
        "estimated weights, nutritional values and allergens. Output ONLY the JSON."
    )

    print(f"\n[{time.strftime('%H:%M:%S')}] Elaborazione in corso: {img_path}...")

    # --- INIZIO CHIAMATA OLLAMA ---
    start_inference = time.time()
    try:
        result = client.ask(
            image_path=full_image_path,
            prompt=prompt,
            system_instruction=system_instruction,
            response_schema=NutritionExtraction,
            examples=few_shot_examples,
        )
    except Exception as e:
        print(f"[ERRORE CRITICO] Il client Ollama è crashato durante ask(): {e}")
        return None

    elapsed = time.time() - start_inference
    print(f"[{time.strftime('%H:%M:%S')}] Inferenza completata in {elapsed:.2f}s")
    # --- FINE CHIAMATA OLLAMA ---

    # --- CONTROLLO CONTENUTO RISULTATO ---
    if result is None:
        print(f"[DEBUG] -> Il risultato è esattamente None.")
        return None

    if isinstance(result, str) and result.strip() == "":
        print(f"[DEBUG] -> Il risultato è una stringa vuota.")
        return None

    print(f"[DEBUG] -> Lunghezza risultato: {len(str(result))} caratteri.")
    print(f"[DEBUG] -> Snippet grezzo (primi 300 caratteri):\n{str(result)[:300]}\n...")

    # --- FASE DI PARSING ---
    try:
        # 1. RIMOSSA LA PARTE DI THINKING (solo se esiste, come in Gemma)
        if "<channel|>" in result:
            # Divide la stringa in due parti usando "<channel|>" e prende l'ultima parte (il JSON)
            result = result.split("<channel|>")[-1]

        # 2. Pulizia standard del markdown (DA ESEGUIRE SEMPRE)
        cleaned_result = result.replace("```json", "").replace("```", "").strip()

        # 3. Validazione
        parsed_result = NutritionExtraction.model_validate_json(cleaned_result)

        prediction_record = {
            "image_path": img_path,
            "food_id": item["food_id"],
            "food_category": item["food_category"],
            "ground_truth_ingredients": item.get("ingredients", []),
            "predicted_nutrition": parsed_result.model_dump()
        }
        print(f"[SUCCESSO] Parsing completato correttamente!")
        return prediction_record

    except Exception as e:
        print(f"[Errore di Parsing] Impossibile analizzare l'output JSON.")
        print(f"Dettagli errore Pydantic: {e}")
        return None


def main():
    print(f"Avvio esecuzione sequenziale con provider: {PROVIDER.upper()}...")

    schema_dict = NutritionExtraction.model_json_schema()

    # 1. System Instruction BASE (Senza token specifici di modello)
    base_system_instruction = (
        "You are an expert nutritionist and food vision analyst. "
        "Your task is to analyze top-down RGB images of dishes to perform a precise nutritional breakdown.\n\n"
        "CORE DIRECTIVES:\n"
        "1. Identify all visible ingredients.\n"
        "2. Deduce logically necessary hidden ingredients (e.g., cooking oils, butter, salt, sauces) based on the preparation method of the visible food.\n"
        "3. Extract all the possibile allergenes (e.g. gluten, dairy products) based on the visible and hidden ingredients extracted before.\n"
        "4. Estimate realistic portion sizes (weight in grams) for each identified item.\n"
        "5. Calculate exact macronutrients based on the estimated weights.\n"
        "6. Sum the total macronutrients accurately.\n\n"
        "You MUST respond ONLY with a valid JSON strictly adhering to this schema:\n"
        f"{json.dumps(schema_dict)}"
    )

    # 2. Configurazione Dinamica del Provider
    system_instruction = base_system_instruction

    if PROVIDER == "gemma":
        model = os.environ.get("OLLAMA_GEMMA_MODEL", "gemma4:e2b-it-q8_0")
        client = GemmaClient(model_name=model)
        # Solo Gemma 4 richiede il token esplicito per il thinking
        system_instruction = "<|think|>\n" + base_system_instruction
        OUTPUT_PREDICTIONS_PATH = "data/prediction/test_predictions_nutrition5k_gemma.json"

    elif PROVIDER == "qwen":
        model = os.environ.get("OLLAMA_QWEN_MODEL", "qwen3.5:4b") # o "qwen3.5:9b"
        client = QwenClient(model_name=model)
        OUTPUT_PREDICTIONS_PATH = "data/prediction/test_predictions_nutrition5k_qwen.json"

    elif PROVIDER == "minicpm":
        model = os.environ.get("OLLAMA_MINICPM_MODEL", "openbmb/minicpm-v4.5:8b")
        # Attiviamo la modalità Deep Thinking come default per questo task complesso
        client = MiniCPMClient(model_name=model, thinking_mode="deep")
        OUTPUT_PREDICTIONS_PATH = "data/prediction/test_predictions_nutrition5k_minicpm.json"

    else:
        raise ValueError(f"Provider '{PROVIDER}' non supportato. Scegli tra 'gemma', 'qwen', 'minicpm'.")

    print(f"[*] Modello selezionato: {model}")
    print(f"[*] File di output: {OUTPUT_PREDICTIONS_PATH}")

    # --- SKELETON ESEMPI FEW-SHOT ---
    example_output = NutritionExtraction(
        visible_ingredients=[
            IngredientWithNutrition(
                name="arugula",
                weight_grams=21.13,
                macronutrients=Macronutrients(calories=5.28, fat=0.15, carbohydrates=0.78, protein=0.55)
            ),
            IngredientWithNutrition(
                name="quinoa",
                weight_grams=211.29,
                macronutrients=Macronutrients(calories=253.55, fat=4.01, carbohydrates=44.37, protein=9.30)
            ),
            IngredientWithNutrition(
                name="white rice",
                weight_grams=190.0,
                macronutrients=Macronutrients(calories=247.0, fat=0.57, carbohydrates=53.20, protein=5.13)
            ),
            IngredientWithNutrition(
                name="parsley",
                weight_grams=4.23,
                macronutrients=Macronutrients(calories=1.57, fat=0.03, carbohydrates=0.27, protein=0.12)
            )
        ],
        hidden_ingredients=[
            IngredientWithNutrition(
                name="olive oil",
                weight_grams=1.06,
                macronutrients=Macronutrients(calories=9.34, fat=1.06, carbohydrates=0.0, protein=0.0)
            ),
            IngredientWithNutrition(
                name="salt",
                weight_grams=2.38,
                macronutrients=Macronutrients(calories=0.0, fat=0.0, carbohydrates=0.0, protein=0.0)
            ),
            IngredientWithNutrition(
                name="garlic",
                weight_grams=5.28,
                macronutrients=Macronutrients(calories=7.87, fat=0.03, carbohydrates=1.74, protein=0.34)
            ),
            IngredientWithNutrition(
                name="shallots",
                weight_grams=1.06,
                macronutrients=Macronutrients(calories=0.21, fat=0.0, carbohydrates=0.03, protein=0.02)
            ),
            IngredientWithNutrition(
                name="thyme",
                weight_grams=1.06,
                macronutrients=Macronutrients(calories=1.07, fat=0.02, carbohydrates=0.25, protein=0.06)
            ),
            IngredientWithNutrition(
                name="lemon juice",
                weight_grams=0.53,
                macronutrients=Macronutrients(calories=0.09, fat=0.0, carbohydrates=0.03, protein=0.0)
            )
        ],
        allergens=["None"],
        total_macronutrients=Macronutrients(
            calories=525.97,
            fat=5.87,
            carbohydrates=100.68,
            protein=15.52
        )
    ).model_dump_json()

    few_shot_examples = [
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1562012076/rgb.png")),
         "output": example_output}
    ]

    os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS_PATH), exist_ok=True)

    # --- CARICAMENTO DATI E FILTRAGGIO ---
    with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if MAX_ITEMS_TO_TEST:
        test_data = test_data[:MAX_ITEMS_TO_TEST]
        print(f"[*] MAX_ITEMS_TO_TEST attivo: esecuzione limitata a {MAX_ITEMS_TO_TEST} elementi.")

    # --- RIPRESA DA CHECKPOINT ---
    existing_predictions = []
    processed_images = set()

    if os.path.exists(OUTPUT_PREDICTIONS_PATH):
        try:
            with open(OUTPUT_PREDICTIONS_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    existing_predictions = json.loads(content)
                    processed_images = {item["image_path"] for item in existing_predictions}
            print(f"Ripresa in corso: trovati {len(processed_images)} elementi già processati.")
        except Exception as e:
            print(f"Errore nella lettura del checkpoint: {e}")

    # --- LOOP DI ESECUZIONE SEQUENZIALE ---
    start_time = time.time()
    successful_count = 0
    total_to_process = len([item for item in test_data if item["image_path"] not in processed_images])

    print(f"\nAvvio loop di inferenza. {total_to_process} immagini rimanenti da processare.")

    try:
        for idx, item in enumerate(test_data, 1):
            record = process_single_image(item, client, system_instruction, few_shot_examples, processed_images)

            if record:
                successful_count += 1
                existing_predictions.append(record)
                processed_images.add(record["image_path"])

                # Logica di salvataggio basata sul numero di elementi
                if successful_count % SAVE_EVERY_N_ITEMS == 0:
                    save_json_safely(existing_predictions, OUTPUT_PREDICTIONS_PATH)
                    print(f"[{successful_count} elementi processati] Checkpoint salvato con successo.")

    except KeyboardInterrupt:
        print("\n[!] Esecuzione interrotta dall'utente (Ctrl+C). Salvataggio di emergenza in corso...")

    finally:
        if existing_predictions:
            save_json_safely(existing_predictions, OUTPUT_PREDICTIONS_PATH)
            print(f"Salvataggio finale completato. Predizioni totali salvate: {len(existing_predictions)}")

        end_time = time.time()
        print(
            f"\nBatch completato! Processate {successful_count} nuove immagini in {end_time - start_time:.2f} secondi.")


if __name__ == "__main__":
    main()