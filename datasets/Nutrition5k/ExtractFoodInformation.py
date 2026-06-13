import json
import os
import time

from OllamaClient.client import OllamaClient
from answer_schema.extraction import Macronutrients, IngredientWithNutrition, NutritionExtraction

PROVIDER = "ollama"

# Checkpoint settings
SAVE_EVERY_N_ITEMS = 5
MAX_ITEMS_TO_TEST = 5

# PATHS
IMAGES_BASE_DIR = "/media/niki/Niki My Passport/scuola/università/magistrale/tesi/dataset/Nutrition5k/nutrition5k_dataset"
TEST_DATASET_PATH = "data/ground truth/test_dataset.json"
OUTPUT_PREDICTIONS_PATH = "data/prediction/test_predictions_nutrition5k_qwen08B.json"


# ==========================================
# --- EXECUTION LOGIC ---
# ==========================================
def save_json_safely(data, filepath):
    """Atomic function to save JSON data safely."""
    temp_filepath = filepath + ".tmp"
    try:
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(temp_filepath, filepath)
    except Exception as e:
        print(f"[ERROR] Unable to save the JSON file: {e}")


def process_single_image(item, client, system_instruction, few_shot_examples, processed_images):
    """Processes a single image sequentially."""
    img_path = item["image_path"]

    if img_path in processed_images:
        return None

    full_image_path = os.path.join(IMAGES_BASE_DIR, img_path.lstrip('/'))

    if not os.path.exists(full_image_path):
        print(f"[Warning] File not found: {full_image_path}")
        return None

    prompt = (
        "This is an overhead RGB image of a dish.\n"
        "Extract the visible and hidden ingredients, estimate their individual weight in grams "
        "and their specific macronutrients, and finally deduce the total macronutrients for the entire dish."
    )

    print(f"Processing: {img_path}...")

    result = client.ask(
        image_path=full_image_path,
        prompt=prompt,
        system_instruction=system_instruction,
        response_schema=NutritionExtraction,
        examples=few_shot_examples,
    )

    if result:
        try:
            cleaned_result = result.replace("```json", "").replace("```", "").strip()
            parsed_result = NutritionExtraction.model_validate_json(cleaned_result)

            prediction_record = {
                "image_path": img_path,
                "food_id": item["food_id"],
                "food_category": item["food_category"],
                "ground_truth_ingredients": item.get("ingredients", []),
                "predicted_nutrition": parsed_result.model_dump()
            }
            return prediction_record

        except Exception as e:
            print(f"[Parse Error] {img_path}: Unable to parse JSON output. Error: {e}")
            return None

    return None


def main():
    print(f"Starting sequential execution with {PROVIDER.upper()} client...")

    schema_dict = NutritionExtraction.model_json_schema()

    system_instruction = (
        "You are an expert nutritionist and food analyst.\n"
        "You must analyze the dish shown in the image from a top-down perspective.\n"
        "You must extract visible and invisible ingredients.\n"
        "You must respond ONLY in valid JSON format respecting the following schema:\n"
        f"{json.dumps(schema_dict)}"
    )

    if PROVIDER == "ollama":
        model = os.environ.get("OLLAMA_QWEN_3.5_0.8B", "qwen3.5:0.8b")
        client = OllamaClient(model_name=model)
    else:
        raise ValueError(f"Provider '{PROVIDER}' not supported.")

    # --- FEW-SHOT EXAMPLES SKELETON ---
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
        {"image_path": os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1562012076/rgb.png"),
         "output": example_output}
    ]

    os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS_PATH), exist_ok=True)

    # --- DATA LOADING AND FILTERING ---
    with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if MAX_ITEMS_TO_TEST:
        test_data = test_data[:MAX_ITEMS_TO_TEST]
        print(f"[*] MAX_ITEMS_TO_TEST enabled: Limiting execution to {MAX_ITEMS_TO_TEST} items.")

    # --- CHECKPOINT RESUMPTION ---
    existing_predictions = []
    processed_images = set()

    if os.path.exists(OUTPUT_PREDICTIONS_PATH):
        try:
            with open(OUTPUT_PREDICTIONS_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    existing_predictions = json.loads(content)
                    processed_images = {item["image_path"] for item in existing_predictions}
            print(f"Resuming progress: Found {len(processed_images)} already processed items.")
        except Exception as e:
            print(f"Error reading checkpoint: {e}")

    # --- SEQUENTIAL EXECUTION LOOP ---
    start_time = time.time()
    successful_count = 0
    total_to_process = len([item for item in test_data if item["image_path"] not in processed_images])

    print(f"\nStarting inference loop. {total_to_process} images left to process.")

    try:
        for idx, item in enumerate(test_data, 1):
            record = process_single_image(item, client, system_instruction, few_shot_examples, processed_images)

            if record:
                successful_count += 1
                existing_predictions.append(record)
                processed_images.add(record["image_path"])

                # Item-based saving logic
                if successful_count % SAVE_EVERY_N_ITEMS == 0:
                    save_json_safely(existing_predictions, OUTPUT_PREDICTIONS_PATH)
                    print(f"[{successful_count} items processed] Checkpoint saved successfully.")

    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user (Ctrl+C). Initiating emergency save...")

    finally:
        if existing_predictions:
            save_json_safely(existing_predictions, OUTPUT_PREDICTIONS_PATH)
            print(f"Final save completed. Total predictions saved: {len(existing_predictions)}")

        end_time = time.time()
        print(f"\nBatch complete! Processed {successful_count} new images in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
