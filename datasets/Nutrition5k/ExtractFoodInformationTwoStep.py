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
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)  # makes prompt_strategies importable also on Kaggle
from prompt_strategies import select_strategy, TWO_STEP_AXES, TWO_STEP_STEP1, TWO_STEP_STEP2

PROVIDER = "gemma"
USE_YOLO = False
DEBUG = False
YOLO_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "yolo_weights", "yolov8s.pt")
_yolo_model = None

# checkpoint settings
SAVE_EVERY_N_ITEMS = 5
MAX_ITEMS_TO_TEST = 3

# paths
IMAGES_BASE_DIR = "/media/niki/Niki My Passport/scuola/universit/magistrale/tesi/dataset/Nutrition5k/nutrition5k_dataset"
TEST_DATASET_PATH = os.path.join(SCRIPT_DIR, "data", "ground truth", "test_dataset.json")
OUTPUT_PREDICTIONS_PATH = os.path.join(SCRIPT_DIR, "data", "prediction", "test_predictions_nutrition5k_twostep.json")


# ==========================================
# --- PYDANTIC SCHEMAS FOR STEP 1 ---
# (Step 2 uses the imported NutritionExtraction)
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
# --- EXECUTION LOGIC ---
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
        print(f"[ERROR] Unable to save the JSON file: {e}")


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
        print(f"[Parsing Error] Pydantic error details: {e}")
        return None


def process_single_image(item, client, step1_sys_prompt, step2_sys_prompt, examples_step1, examples_step2,
                         processed_images, step1_cfg, step2_cfg):
    img_path = item["image_path"]

    if img_path in processed_images:
        return None

    full_image_path = normalize_path(os.path.join(IMAGES_BASE_DIR, img_path))

    if not os.path.exists(full_image_path):
        print(f"[Warning] File not found: {full_image_path}")
        return None

    print(f"\n[{time.strftime('%H:%M:%S')}] Processing: {img_path}...")

    if USE_YOLO:
        print(f"[YOLO] Running inference on {img_path}...")
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
                    raw_boxes.append({"class_name": class_name, "coords": coords, "conf": conf, "class_id": class_id, "weight": conf, "count": 1})
                
                print(f"[YOLO ORIGINAL] Detected ingredients: {[b['class_name'] for b in raw_boxes]}")
                print(f"[YOLO ORIGINAL] Confidences: {[round(b['conf'], 2) for b in raw_boxes]}")
                
                # Group overlapping boxes and compute the weighted average (grouped WBF)
                IOU_THRESHOLD = 0.5
                filtered_boxes = []
                
                boxes_by_class = {}
                for box in raw_boxes:
                    cname = box["class_name"]
                    if cname not in boxes_by_class:
                        boxes_by_class[cname] = []
                    boxes_by_class[cname].append(box)
                    
                for cname, class_boxes in boxes_by_class.items():
                    groups = []
                    for box in class_boxes:
                        matching_groups_indices = []
                        for i, group in enumerate(groups):
                            overlaps = False
                            for g_box in group:
                                if calculate_iou(box["coords"], g_box["coords"]) > IOU_THRESHOLD:
                                    overlaps = True
                                    break
                            if overlaps:
                                matching_groups_indices.append(i)
                        
                        if not matching_groups_indices:
                            groups.append([box])
                        else:
                            new_group = [box]
                            for i in sorted(matching_groups_indices, reverse=True):
                                new_group.extend(groups.pop(i))
                            groups.append(new_group)
                            
                    for group in groups:
                        total_weight = sum(b["weight"] for b in group)
                        avg_coords = [0.0, 0.0, 0.0, 0.0]
                        for b in group:
                            w = b["weight"]
                            c = b["coords"]
                            avg_coords[0] += c[0] * w
                            avg_coords[1] += c[1] * w
                            avg_coords[2] += c[2] * w
                            avg_coords[3] += c[3] * w
                            
                        avg_coords = [x / total_weight for x in avg_coords]
                        
                        filtered_boxes.append({
                            "class_name": cname,
                            "class_id": group[0]["class_id"],
                            "coords": avg_coords,
                            "conf": total_weight / len(group),
                            "weight": total_weight,
                            "count": len(group)
                        })
                
                # Sort the filtered boxes by descending confidence
                filtered_boxes.sort(key=lambda x: x["conf"], reverse=True)
                
                detected_ingredients = [b["class_name"] for b in filtered_boxes]
                bounding_boxes = [b["coords"] for b in filtered_boxes]
                detected_confidences = [round(b["conf"], 2) for b in filtered_boxes]
                    
                print(f"[YOLO FILTERED] Detected ingredients ({len(detected_ingredients)}): {detected_ingredients}")
                print(f"[YOLO FILTERED] Bounding box coordinates: {bounding_boxes}")
                print(f"[YOLO FILTERED] Bounding box confidences: {detected_confidences}")
                
                # --- VISUALIZATION OF THE NEW BOUNDING BOXES ---
                if DEBUG:
                    from ultralytics.utils.plotting import colors, Annotator
                    annotator = Annotator(img_resized.copy(), line_width=2, example=str(_yolo_model.names))
                    for idx, box in enumerate(filtered_boxes, start=1):
                        coords = box["coords"]
                        class_id = box["class_id"]
                        
                        # Build the label with only an incremental ID, without confidence
                        label = f"{idx}"
                        # box_label also takes care of assigning the YOLO colors
                        annotator.box_label(coords, label, color=colors(class_id, True))
                    
                    drawn_img = annotator.result()
                    cv2.imshow("Modified Bounding Boxes", drawn_img)
                    cv2.waitKey()  # Show the image (press any key to continue)
                    cv2.destroyAllWindows()
                # --------------------------------------------------
            else:
                print(f"[Warning] Unable to read the image with cv2: {full_image_path}")
        except Exception as e:
            print(f"[YOLO ERROR] {e}")
        return None  # Skip the LLM step

    # ==========================
    # STEP 1: Visual Analysis (weights and ingredients only)
    # ==========================
    prompt1 = step1_cfg["prompt"]

    print(f"[{time.strftime('%H:%M:%S')}] Running Step 1 (Visual Analysis)...")
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
        print(f"[CRITICAL ERROR] Crash during Step 1: {e}")
        return None

    print(f"[{time.strftime('%H:%M:%S')}] Step 1 completed in {time.time() - start_step1:.2f}s")

    parsed_step1 = parse_llm_json(result_step1, FoodAnalysisStep1)
    if not parsed_step1:
        print(f"[ERROR] Failed to parse Step 1.")
        return None

    # ==========================
    # STEP 2: Nutrition Analysis (adds macronutrients)
    # ==========================
    prompt2 = step2_cfg["prompt"].replace("{step1_json}", parsed_step1.model_dump_json())

    print(f"[{time.strftime('%H:%M:%S')}] Running Step 2 (Nutrition Analysis)...")
    start_step2 = time.time()
    try:
        result_step2 = client.ask(
            image_path=full_image_path,  # The image is passed for robustness, as suggested in the paper
            prompt=prompt2,
            system_instruction=step2_sys_prompt,
            response_schema=NutritionExtraction,
            examples=examples_step2,
        )
    except Exception as e:
        print(f"[CRITICAL ERROR] Crash during Step 2: {e}")
        return None

    print(f"[{time.strftime('%H:%M:%S')}] Step 2 completed in {time.time() - start_step2:.2f}s")

    parsed_step2 = parse_llm_json(result_step2, NutritionExtraction)
    if not parsed_step2:
        print(f"[ERROR] Failed to parse Step 2.")
        return None

    prediction_record = {
        "image_path": img_path,
        "food_id": item["food_id"],
        "food_category": item["food_category"],
        "ground_truth_ingredients": item.get("ingredients", []),
        "predicted_food_analysis_step1": parsed_step1.model_dump(),
        "predicted_nutrition_step2": parsed_step2.model_dump()
    }
    print(f"[SUCCESS] Two-step process completed successfully!")
    return prediction_record


def main(step1=None, step2=None):
    print(f"Starting sequential execution with provider: {PROVIDER.upper()} (TWO-STEP CUSTOM PROMPTING)...")

    # --- 1. SYSTEM INSTRUCTIONS (chosen at startup, see prompt_strategies.py) ---
    choice = select_strategy(TWO_STEP_AXES, overrides={"step1": step1, "step2": step2},
                             title="Prompt strategy (two step)")
    step1_cfg = TWO_STEP_STEP1[choice["step1"]]
    step2_cfg = TWO_STEP_STEP2[choice["step2"]]
    base_step1_instruction = step1_cfg["system"]
    base_step2_instruction = step2_cfg["system"]

    # --- 2. PROVIDER CONFIGURATION ---
    step1_sys_prompt = base_step1_instruction
    step2_sys_prompt = base_step2_instruction

    global OUTPUT_PREDICTIONS_PATH

    if PROVIDER == "gemma":
        model = os.environ.get("OLLAMA_GEMMA", "gemma4:e2b")
        client = GemmaClient(model_name=model)
        step1_sys_prompt = "<|think|>\n" + base_step1_instruction
        step2_sys_prompt = "<|think|>\n" + base_step2_instruction
        OUTPUT_PREDICTIONS_PATH = os.path.join(SCRIPT_DIR, "data", "prediction", "test_predictions_nutrition5k_twostep_gemma.json")

    elif PROVIDER == "qwen":
        model = os.environ.get("OLLAMA_QWEN_MODEL", "qwen3.5:4b")
        client = QwenClient(model_name=model)
        OUTPUT_PREDICTIONS_PATH = os.path.join(SCRIPT_DIR, "data", "prediction", "test_predictions_nutrition5k_twostep_qwen.json")

    elif PROVIDER == "minicpm":
        model = os.environ.get("OLLAMA_MINICPM_MODEL", "openbmb/minicpm-v4.5:8b")
        client = MiniCPMClient(model_name=model, thinking_mode="deep")
        OUTPUT_PREDICTIONS_PATH = os.path.join(SCRIPT_DIR, "data", "prediction", "test_predictions_nutrition5k_twostep_minicpm.json")

    else:
        raise ValueError(f"Provider '{PROVIDER}' is not supported.")

    # Keep results of different strategies in different files (checkpoint resume is path based)
    if (choice["step1"], choice["step2"]) != ("weights_only", "nutrition"):
        OUTPUT_PREDICTIONS_PATH = OUTPUT_PREDICTIONS_PATH.replace(".json", f"_{choice['step1']}_{choice['step2']}.json")

    print(f"[*] Selected model: {model}")
    print(f"[*] Output file: {OUTPUT_PREDICTIONS_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS_PATH), exist_ok=True)

    # --- 3. FEW-SHOT EXAMPLES SPLIT BY STEP ---
    # Example Step 1 output (weights only)
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

    # Example Step 2 output (full macronutrients)
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

    # --- DATA LOADING ---
    with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if MAX_ITEMS_TO_TEST:
        test_data = test_data[:MAX_ITEMS_TO_TEST]

    # --- CHECKPOINT RESUME ---
    existing_predictions = []
    processed_images = set()

    if os.path.exists(OUTPUT_PREDICTIONS_PATH):
        try:
            with open(OUTPUT_PREDICTIONS_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    existing_predictions = json.loads(content)
                    processed_images = {item["image_path"] for item in existing_predictions}
            print(f"Resuming: found {len(processed_images)} already processed items.")
        except Exception as e:
            print(f"Error reading the checkpoint: {e}")

    # --- SEQUENTIAL INFERENCE ---
    start_time = time.time()
    successful_count = 0
    total_to_process = len([item for item in test_data if item["image_path"] not in processed_images])

    print(f"\nStarting inference loop. {total_to_process} images left to process.")

    try:
        for idx, item in enumerate(test_data, 1):
            record = process_single_image(
                item, client, step1_sys_prompt, step2_sys_prompt, examples_step1, examples_step2, processed_images,
                step1_cfg, step2_cfg
            )

            if record:
                successful_count += 1
                existing_predictions.append(record)
                processed_images.add(record["image_path"])

                if successful_count % SAVE_EVERY_N_ITEMS == 0:
                    save_json_safely(existing_predictions, OUTPUT_PREDICTIONS_PATH)
                    print(f"[{successful_count} items processed] Checkpoint saved.")

    except KeyboardInterrupt:
        print("\n[!] Execution interrupted. Emergency save...")

    finally:
        if existing_predictions:
            save_json_safely(existing_predictions, OUTPUT_PREDICTIONS_PATH)
            print(f"Final save completed. Predictions saved: {len(existing_predictions)}")

        end_time = time.time()
        print(
            f"\nBatch completed! Processed {successful_count} new images in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()