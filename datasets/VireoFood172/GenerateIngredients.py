import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from GoogleLLMClient.client import GoogleClient
from HuggingFaceLLMClient.client import HuggingFaceClient
from OpenRouterLLMClient.client import OpenRouterClient
from answer_schema.extraction import IngredientExtraction

# --- CONFIGURATION ---
PROVIDER = "google"
MAX_CONCURRENT_WORKERS = 2

MAX_RPM = 8
DELAY_BETWEEN_REQUESTS = 60.0 / MAX_RPM

ONE_IMAGE_PER_CLASS = True
MAX_ITEMS_TO_TEST = None

# --- CONFIGURATION PATHS ---
IMAGES_BASE_DIR = "/media/niki/Niki My Passport/scuola/università/magistrale/tesi/dataset/VireoFood172/VireoFood172/ready_chinese_food"
TEST_DATASET_PATH = "data/ground truth/test_dataset.json"
OUTPUT_PREDICTIONS_PATH = "data/prediction/test_predictions_np_gemma31B.json"

# --- LOCKS E VARIABILI GLOBALI ---
file_write_lock = Lock()
rate_limit_lock = Lock()
last_api_call_time = [0.0]


def save_json_safely(data, filepath):
    """Atomic function to save LLM answer in a JSON file."""
    temp_filepath = filepath + ".tmp"
    try:
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(temp_filepath, filepath)
    except Exception as e:
        print(f"[ERROR] Impossibile to save the JSON file: {e}")


def process_single_image(item, client, system_instruction, few_shot_examples, processed_images):
    """Worker function to process a single image. Designed to run in parallel."""
    img_path = item["image_path"]

    if img_path in processed_images:
        return None

    full_image_path = os.path.join(IMAGES_BASE_DIR, img_path.lstrip('/'))

    if not os.path.exists(full_image_path):
        print(f"[Warning] File not found: {full_image_path}")
        return None

    prompt = (
        f"This is an image of a dish.\n"
        "Extract the ingredients and deduce the allergens from this dish."
    )

    with rate_limit_lock:
        current_time = time.time()
        elapsed = current_time - last_api_call_time[0]
        if elapsed < DELAY_BETWEEN_REQUESTS:
            time.sleep(DELAY_BETWEEN_REQUESTS - elapsed)
        last_api_call_time[0] = time.time()

    result = client.ask(
        image_path=full_image_path,
        prompt=prompt,
        system_instruction=system_instruction,
        response_schema=IngredientExtraction,
        examples=few_shot_examples,
    )

    if result:
        try:
            cleaned_result = result.replace("```json", "").replace("```", "").strip()
            parsed_result = IngredientExtraction.model_validate_json(cleaned_result)

            prediction_record = {
                "image_path": img_path,
                "food_id": item["food_id"],
                "food_category": item["food_category"],
                "ground_truth_ingredients": item["ingredients"],
                "predicted_ingredients": parsed_result.model_dump()
            }
            return prediction_record

        except Exception as e:
            print(f"[Parse Error] {img_path}: {e}")
            return None

    return None


def main():
    print(f"Starting with {PROVIDER.upper()} client using Multithreading.")

    schema_dict = IngredientExtraction.model_json_schema()

    system_instruction = (
        "You are a chef, expert food critic and food analyst.\n"
        "You must respond ONLY in valid JSON format respecting the following schema:\n"
        f"{json.dumps(schema_dict)}"
    )

    # --- CLIENT INITIALIZATION ---
    if PROVIDER == "google":
        model = os.environ.get("GOOGLE_GEMMA_31B", "gemini-1.5-pro")
        client = GoogleClient(model_name=model)

    elif PROVIDER == "openrouter":
        model = os.environ.get("OPENROUTER_MINIMAX_M2.5")
        client = OpenRouterClient(model_name=model)

    elif PROVIDER == "huggingface":
        model = os.environ.get("HUGGINGFACE_QWEN_3.5_27B")
        client = HuggingFaceClient(model_name=model, max_retries=5, base_delay=2.0)

    else:
        raise ValueError(f"Provider '{PROVIDER}' not supported.")

    # --- FEW-SHOT EXAMPLES ---
    example_output = IngredientExtraction(
        visible_ingredients=["tomato sauce", "mozzarella cheese", "basil"],
        hidden_ingredients=["flour", "water", "salt", "yeast", "olive oil"],
        allergens=["gluten", "dairy"]
    ).model_dump_json()

    few_shot_examples = [
        {"image_path": "/home/niki/Scrivania/margherita.jpg", "output": example_output}
    ]

    os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS_PATH), exist_ok=True)

    # --- LOAD DATA ---
    with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # --- FILTRO IMMAGINE PER CLASSE ---
    if ONE_IMAGE_PER_CLASS:
        filtered_data = []
        seen_classes = set()

        for item in test_data:
            food_id = item.get("food_id")
            if food_id not in seen_classes:
                filtered_data.append(item)
                seen_classes.add(food_id)

        test_data = filtered_data
        print(f"[*] ONE_IMAGE_PER_CLASS attivatato: Dataset ridotto a {len(test_data)} piatti unici.")

    if MAX_ITEMS_TO_TEST:
        test_data = test_data[:MAX_ITEMS_TO_TEST]

    existing_predictions = []
    processed_images = set()

    if os.path.exists(OUTPUT_PREDICTIONS_PATH):
        try:
            with open(OUTPUT_PREDICTIONS_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    existing_predictions = json.loads(content)
                    processed_images = {item["image_path"] for item in existing_predictions}
            print(f"Resuming progress: Found {len(processed_images)} processed items.")
        except Exception as e:
            print(f"Error reading checkpoint: {e}")

    # --- CHIAMATE PARALLELE ---
    print(f"Launching ThreadPoolExecutor with {MAX_CONCURRENT_WORKERS} workers...")
    print(f"Applying Rate Limit: Max {MAX_RPM} RPM (Delay: {DELAY_BETWEEN_REQUESTS:.2f}s per request)")

    start_time = time.time()
    successful_count = 0

    try:
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:


            futures = {
                executor.submit(
                    process_single_image, item, client, system_instruction, few_shot_examples, processed_images
                ): item
                for item in test_data
            }

            for idx, future in enumerate(as_completed(futures), 1):
                record = future.result()

                if record:
                    successful_count += 1
                    with file_write_lock:
                        existing_predictions.append(record)
                        processed_images.add(record["image_path"])

                        if successful_count % 5 == 0:
                            save_json_safely(existing_predictions, OUTPUT_PREDICTIONS_PATH)
                            print(f"[{successful_count} done] Checkpoint saved...")

    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user (Ctrl+C). Performing emergency save...")

    finally:
        with file_write_lock:
            if existing_predictions:
                save_json_safely(existing_predictions, OUTPUT_PREDICTIONS_PATH)
                print(f"Final save complete. Total predictions stored: {len(existing_predictions)}")

    end_time = time.time()
    print(f"\nBatch complete! Processed {successful_count} new images in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()