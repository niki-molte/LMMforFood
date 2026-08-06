import json
import os
import time
import unicodedata
from pydantic import BaseModel, Field

from OllamaClient.gemma_client import GemmaClient
from OllamaClient.qwen_client import QwenClient
from OllamaClient.miniCPM_client import MiniCPMClient
from answer_schema.extraction import Macronutrients, IngredientWithNutrition, NutritionExtraction

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

PROVIDER = "minicpm"
USE_YOLO = True
DEBUG = False
YOLO_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "yolo_weights", "yolov8s.pt")
_yolo_model = None

# impostazioni checkpoint
SAVE_EVERY_N_ITEMS = 5
MAX_ITEMS_TO_TEST = 3

# percorsi
IMAGES_BASE_DIR = "/media/niki/Niki My Passport/scuola/universit/magistrale/tesi/dataset/Nutrition5k/nutrition5k_dataset"
<<<<<<< Updated upstream
TEST_DATASET_PATH = "datasets/Nutrition5k/data/ground truth/test_dataset.json"
OUTPUT_PREDICTIONS_PATH = "data/prediction/test_predictions_nutrition5k_twostep_custom.json"
=======
TEST_DATASET_PATH = os.path.join(SCRIPT_DIR, "data", "ground truth", "test_dataset.json")
OUTPUT_PREDICTIONS_PATH = os.path.join(SCRIPT_DIR, "data", "prediction", "test_predictions_nutrition5k_twostep.json")
>>>>>>> Stashed changes


# ==========================================
# --- SCHEMI PYDANTIC PER STEP 1 ---
# (Step 2 userà il tuo NutritionExtraction importato)
# ==========================================
class BaseIngredient(BaseModel):
    name: str = Field(description="Name of the ingredient")
    weight_grams: float = Field(description="Estimated weight in grams")


class FoodAnalysisStep1(BaseModel):
    visible_ingredients: list[BaseIngredient] = Field(
        description="List of identified visible ingredients and their weights.")
    hidden_ingredients: list[BaseIngredient] = Field(
        description="List of deduced hidden ingredients (e.g., oils, butter, salt) and their weights.")
    allergens: list[str] = Field(description="List of possible allergens based on the ingredients.")


# ==========================================
# --- LOGICA DI ESECUZIONE ---
# ==========================================
def normalize_path(path: str) -> str:
    return unicodedata.normalize("NFC", path)


def save_json_safely(data, filepath):
    temp_filepath = filepath + ".tmp"
    try:
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(temp_filepath, filepath)
    except Exception as e:
        print(f"[ERRORE] Impossibile salvare il file JSON: {e}")


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denominator = float(boxAArea + boxBArea - interArea)
    if denominator == 0:
        return 0.0
    return interArea / denominator


def parse_llm_json(result_str, schema_model):
    if not result_str or not str(result_str).strip():
        return None
    try:
        if "<channel|>" in result_str:
            result_str = result_str.split("<channel|>")[-1]
        cleaned_result = result_str.replace("```json", "").replace("```", "").strip()
        parsed_result = schema_model.model_validate_json(cleaned_result)
        return parsed_result
    except Exception as e:
        print(f"[Errore di Parsing] Dettagli errore Pydantic: {e}")
        return None


def process_single_image(item, client, step1_sys_prompt, step2_sys_prompt, examples_step1, examples_step2,
                         processed_images):
    img_path = item["image_path"]

    if img_path in processed_images:
        return None

    full_image_path = normalize_path(os.path.join(IMAGES_BASE_DIR, img_path))

    if not os.path.exists(full_image_path):
        print(f"[Avviso] File non trovato: {full_image_path}")
        return None

    print(f"\n[{time.strftime('%H:%M:%S')}] Elaborazione in corso: {img_path}...")

    if USE_YOLO:
        print(f"[YOLO] Esecuzione inferenza su {img_path}...")
        try:
            import cv2
            from ultralytics import YOLO
            global _yolo_model
            if _yolo_model is None:
                _yolo_model = YOLO(YOLO_WEIGHTS_PATH)
            
            img = cv2.imread(full_image_path)
            if img is not None:
                img_resized = cv2.resize(img, (640, 640))
                results = _yolo_model(img_resized)
                if DEBUG:
                    results[0].show()
                
                raw_boxes = []
                for box in results[0].boxes:
                    class_id = int(box.cls[0].item())
                    class_name = _yolo_model.names[class_id]
                    coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0].item()) if hasattr(box, 'conf') and box.conf is not None else 1.0
                    raw_boxes.append({"class_name": class_name, "coords": coords, "conf": conf, "class_id": class_id})
                
                print(f"[YOLO ORIGINALE] Ingredienti rilevati: {[b['class_name'] for b in raw_boxes]}")
                print(f"[YOLO ORIGINALE] Confidenze: {[round(b['conf'], 2) for b in raw_boxes]}")
                
                # Questa cosa si fa solo se ci sono ingredienti uguali rilevati
                raw_boxes.sort(key=lambda x: x["conf"], reverse=True)
                filtered_boxes = []
                IOU_THRESHOLD = 0.5
                
                for current_box in raw_boxes:
                    should_add = True
                    for i, kept_box in enumerate(filtered_boxes):
                        if current_box["class_name"] == kept_box["class_name"]:
                            # Trovato un ingrediente uguale, controlliamo l'IoU
                            iou_val = calculate_iou(current_box["coords"], kept_box["coords"])
                            if iou_val > IOU_THRESHOLD:
                                # Unisci le label identiche se la IoU è alta
                                k_coords = kept_box["coords"]
                                c_coords = current_box["coords"]
                                filtered_boxes[i]["coords"] = [
                                    min(k_coords[0], c_coords[0]),
                                    min(k_coords[1], c_coords[1]),
                                    max(k_coords[2], c_coords[2]),
                                    max(k_coords[3], c_coords[3])
                                ]
                                # Essendo state unite, non aggiungiamo la box separatamente
                                should_add = False
                                break
                            
                    if should_add:
                        filtered_boxes.append(current_box)
                
                detected_ingredients = [b["class_name"] for b in filtered_boxes]
                bounding_boxes = [b["coords"] for b in filtered_boxes]
                detected_confidences = [round(b["conf"], 2) for b in filtered_boxes]
                    
                print(f"[YOLO FILTRATO] Ingredienti rilevati ({len(detected_ingredients)}): {detected_ingredients}")
                print(f"[YOLO FILTRATO] Coordinate Bounding Box: {bounding_boxes}")
                print(f"[YOLO FILTRATO] Confidenze Bounding Box: {detected_confidences}")
                
                # --- VISUALIZZAZIONE DELLE NUOVE BOUNDING BOX ---
                if DEBUG:
                    from ultralytics.utils.plotting import colors, Annotator
                    annotator = Annotator(img_resized.copy(), line_width=2, example=str(_yolo_model.names))
                    for b in filtered_boxes:
                        c = b["coords"]
                        color = colors(b["class_id"], True)
                        label = f'{b["class_name"]} {b["conf"]:.2f}'
                        annotator.box_label(c, label, color=color)
                    
                    drawn_img = annotator.result()
                    cv2.imshow("Modified Bounding Boxes", drawn_img)
                    cv2.waitKey()  # Mostra l'immagine (premi un tasto per continuare)
                    cv2.destroyAllWindows()
                # --------------------------------------------------
            else:
                print(f"[Avviso] Impossibile leggere l'immagine con cv2: {full_image_path}")
        except Exception as e:
            print(f"[ERRORE YOLO] {e}")
        return None  # Salta il passaggio al LLM per il momento

    # ==========================
    # STEP 1: Analisi Visiva (Solo pesi e ingredienti)
    # ==========================
    prompt1 = "Analyze this dish. Provide the full breakdown of visible/hidden ingredients, estimated weights, and allergens. Output ONLY the JSON."

    print(f"[{time.strftime('%H:%M:%S')}] Esecuzione Step 1 (Visual Analysis)...")
    start_step1 = time.time()
    try:
        result_step1 = client.ask(
            image_path=full_image_path,
            prompt=prompt1,
            system_instruction=step1_sys_prompt,
            response_schema=FoodAnalysisStep1,
            examples=examples_step1,
        )
    except Exception as e:
        print(f"[ERRORE CRITICO] Crash durante Step 1: {e}")
        return None

    print(f"[{time.strftime('%H:%M:%S')}] Step 1 completato in {time.time() - start_step1:.2f}s")

    parsed_step1 = parse_llm_json(result_step1, FoodAnalysisStep1)
    if not parsed_step1:
        print(f"[ERRORE] Fallito il parsing dello Step 1.")
        return None

    # ==========================
    # STEP 2: Analisi Nutrizionale (Aggiunta Macronutrienti)
    # ==========================
    prompt2 = (
        "Analyze this dish with the following extracted details. Calculate exact macronutrients based on the estimated weights.\n\n"
        "Extracted Food Details:\n"
        f"{parsed_step1.model_dump_json()}\n\n"
        "Output ONLY the final JSON."
    )

    print(f"[{time.strftime('%H:%M:%S')}] Esecuzione Step 2 (Nutrition Analysis)...")
    start_step2 = time.time()
    try:
        result_step2 = client.ask(
            image_path=full_image_path,  # Passiamo l'immagine per robustezza, come suggerito nel paper
            prompt=prompt2,
            system_instruction=step2_sys_prompt,
            response_schema=NutritionExtraction,
            examples=examples_step2,
        )
    except Exception as e:
        print(f"[ERRORE CRITICO] Crash durante Step 2: {e}")
        return None

    print(f"[{time.strftime('%H:%M:%S')}] Step 2 completato in {time.time() - start_step2:.2f}s")

    parsed_step2 = parse_llm_json(result_step2, NutritionExtraction)
    if not parsed_step2:
        print(f"[ERRORE] Fallito il parsing dello Step 2.")
        return None

    prediction_record = {
        "image_path": img_path,
        "food_id": item["food_id"],
        "food_category": item["food_category"],
        "ground_truth_ingredients": item.get("ingredients", []),
        "predicted_food_analysis_step1": parsed_step1.model_dump(),
        "predicted_nutrition_step2": parsed_step2.model_dump()
    }
    print(f"[SUCCESSO] Processo a due step completato correttamente!")
    return prediction_record


def main():
    print(f"Avvio esecuzione sequenziale con provider: {PROVIDER.upper()} (TWO-STEP CUSTOM PROMPTING)...")

    # --- 1. SYSTEM INSTRUCTIONS ---
    base_step1_instruction = (
        "You are an expert food vision analyst.\n"
        "Your task is to analyze top-down RGB images of dishes to identify ingredients and portions.\n\n"
        "CORE DIRECTIVES FOR STEP 1:\n"
        "1. Identify all visible ingredients.\n"
        "2. Deduce logically necessary hidden ingredients (e.g., cooking oils, butter, salt, sauces).\n"
        "3. Extract all possible allergens based on visible and hidden ingredients.\n"
        "4. Estimate realistic portion sizes (weight in grams) for each item.\n"
        "5. DO NOT calculate macronutrients. Focus ONLY on identification and weight estimation.\n"
        "You MUST respond ONLY with a valid JSON."
    )

    base_step2_instruction = (
        "You are an expert nutritionist.\n"
        "Your task is to perform a precise nutritional breakdown based on the provided ingredient list and weights.\n\n"
        "CORE DIRECTIVES FOR STEP 2:\n"
        "1. Calculate exact macronutrients (calories, fat, carbohydrates, protein) for each provided ingredient.\n"
        "You MUST respond ONLY with a valid JSON."
    )

    # --- 2. CONFIGURAZIONE PROVIDER ---
    step1_sys_prompt = base_step1_instruction
    step2_sys_prompt = base_step2_instruction

    global OUTPUT_PREDICTIONS_PATH

    if PROVIDER == "gemma":
        model = os.environ.get("OLLAMA_GEMMA_MODEL", "gemma4:e4b-it-q8_0")
        client = GemmaClient(model_name=model)
        step1_sys_prompt = "<|think|>\n" + base_step1_instruction
        step2_sys_prompt = "<|think|>\n" + base_step2_instruction
<<<<<<< Updated upstream
        OUTPUT_PREDICTIONS_PATH = "data/prediction/test_predictions_nutrition5k_twostep_custom_gemma.json"
=======
        OUTPUT_PREDICTIONS_PATH = os.path.join(SCRIPT_DIR, "data", "prediction", "test_predictions_nutrition5k_twostep_gemma.json")
>>>>>>> Stashed changes

    elif PROVIDER == "qwen":
        model = os.environ.get("OLLAMA_QWEN_MODEL", "qwen3.5:4b")
        client = QwenClient(model_name=model)
<<<<<<< Updated upstream
        OUTPUT_PREDICTIONS_PATH = "data/prediction/test_predictions_nutrition5k_twostep_custom_qwen.json"
=======
        OUTPUT_PREDICTIONS_PATH = os.path.join(SCRIPT_DIR, "data", "prediction", "test_predictions_nutrition5k_twostep_qwen.json")
>>>>>>> Stashed changes

    elif PROVIDER == "minicpm":
        model = os.environ.get("OLLAMA_MINICPM_MODEL", "openbmb/minicpm-v4.5:8b")
        client = MiniCPMClient(model_name=model, thinking_mode="deep")
<<<<<<< Updated upstream
        OUTPUT_PREDICTIONS_PATH = "data/prediction/test_predictions_nutrition5k_twostep_custom_minicpm.json"
=======
        OUTPUT_PREDICTIONS_PATH = os.path.join(SCRIPT_DIR, "data", "prediction", "test_predictions_nutrition5k_twostep_minicpm.json")
>>>>>>> Stashed changes

    else:
        raise ValueError(f"Provider '{PROVIDER}' non supportato.")

    print(f"[*] Modello selezionato: {model}")
    print(f"[*] File di output: {OUTPUT_PREDICTIONS_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS_PATH), exist_ok=True)

    # --- 3. FEW-SHOT EXAMPLES DIVISI PER STEP ---
    # Esempio Output Step 1 (Solo Pesi)
    example_step1_output = FoodAnalysisStep1(
        visible_ingredients=[
            BaseIngredient(name="arugula", weight_grams=21.13),
            BaseIngredient(name="quinoa", weight_grams=211.29),
            BaseIngredient(name="white rice", weight_grams=190.0),
            BaseIngredient(name="parsley", weight_grams=4.23)
        ],
        hidden_ingredients=[
            BaseIngredient(name="olive oil", weight_grams=1.06),
            BaseIngredient(name="salt", weight_grams=2.38),
            BaseIngredient(name="garlic", weight_grams=5.28),
            BaseIngredient(name="shallots", weight_grams=1.06),
            BaseIngredient(name="thyme", weight_grams=1.06),
            BaseIngredient(name="lemon juice", weight_grams=0.53)
        ],
        allergens=["None"]
    ).model_dump_json()

    # Esempio Output Step 2 (Macronutrienti completi)
    example_step2_output = NutritionExtraction(
        visible_ingredients=[
            IngredientWithNutrition(name="arugula", weight_grams=21.13,
                                    macronutrients=Macronutrients(calories=5.28, fat=0.15, carbohydrates=0.78,
                                                                  protein=0.55)),
            IngredientWithNutrition(name="quinoa", weight_grams=211.29,
                                    macronutrients=Macronutrients(calories=253.55, fat=4.01, carbohydrates=44.37,
                                                                  protein=9.30)),
            IngredientWithNutrition(name="white rice", weight_grams=190.0,
                                    macronutrients=Macronutrients(calories=247.0, fat=0.57, carbohydrates=53.20,
                                                                  protein=5.13)),
            IngredientWithNutrition(name="parsley", weight_grams=4.23,
                                    macronutrients=Macronutrients(calories=1.57, fat=0.03, carbohydrates=0.27,
                                                                  protein=0.12))
        ],
        hidden_ingredients=[
            IngredientWithNutrition(name="olive oil", weight_grams=1.06,
                                    macronutrients=Macronutrients(calories=9.34, fat=1.06, carbohydrates=0.0,
                                                                  protein=0.0)),
            IngredientWithNutrition(name="salt", weight_grams=2.38,
                                    macronutrients=Macronutrients(calories=0.0, fat=0.0, carbohydrates=0.0,
                                                                  protein=0.0)),
            IngredientWithNutrition(name="garlic", weight_grams=5.28,
                                    macronutrients=Macronutrients(calories=7.87, fat=0.03, carbohydrates=1.74,
                                                                  protein=0.34)),
            IngredientWithNutrition(name="shallots", weight_grams=1.06,
                                    macronutrients=Macronutrients(calories=0.21, fat=0.0, carbohydrates=0.03,
                                                                  protein=0.02)),
            IngredientWithNutrition(name="thyme", weight_grams=1.06,
                                    macronutrients=Macronutrients(calories=1.07, fat=0.02, carbohydrates=0.25,
                                                                  protein=0.06)),
            IngredientWithNutrition(name="lemon juice", weight_grams=0.53,
                                    macronutrients=Macronutrients(calories=0.09, fat=0.0, carbohydrates=0.03,
                                                                  protein=0.0))
        ],
        allergens=["None"],
        total_macronutrients=Macronutrients(calories=525.97, fat=5.87, carbohydrates=100.68, protein=15.52)
    ).model_dump_json()

    example_path = normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1562012076/rgb.png"))

    examples_step1 = [{"image_path": example_path, "output": example_step1_output}]
    examples_step2 = [{"image_path": example_path, "output": example_step2_output}]

    # --- CARICAMENTO DATI ---
    with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if MAX_ITEMS_TO_TEST:
        test_data = test_data[:MAX_ITEMS_TO_TEST]

    # --- RIPRESA CHECKPOINT ---
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

    # --- INFERENZA SEQUENZIALE ---
    start_time = time.time()
    successful_count = 0
    total_to_process = len([item for item in test_data if item["image_path"] not in processed_images])

    print(f"\nAvvio loop di inferenza. {total_to_process} immagini rimanenti da processare.")

    try:
        for idx, item in enumerate(test_data, 1):
            record = process_single_image(
                item, client, step1_sys_prompt, step2_sys_prompt, examples_step1, examples_step2, processed_images
            )

            if record:
                successful_count += 1
                existing_predictions.append(record)
                processed_images.add(record["image_path"])

                if successful_count % SAVE_EVERY_N_ITEMS == 0:
                    save_json_safely(existing_predictions, OUTPUT_PREDICTIONS_PATH)
                    print(f"[{successful_count} elementi processati] Checkpoint salvato.")

    except KeyboardInterrupt:
        print("\n[!] Esecuzione interrotta. Salvataggio di emergenza...")

    finally:
        if existing_predictions:
            save_json_safely(existing_predictions, OUTPUT_PREDICTIONS_PATH)
            print(f"Salvataggio finale completato. Predizioni salvate: {len(existing_predictions)}")

        end_time = time.time()
        print(
            f"\nBatch completato! Processate {successful_count} nuove immagini in {end_time - start_time:.2f} secondi.")


if __name__ == "__main__":
    main()