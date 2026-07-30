import json
import os
import time
import unicodedata
from pydantic import BaseModel, Field

from OllamaClient.gemma_client import GemmaClient
from OllamaClient.qwen_client import QwenClient
from OllamaClient.miniCPM_client import MiniCPMClient
from answer_schema.extraction import Macronutrients, IngredientWithNutrition, NutritionExtraction

PROVIDER = "minicpm"

# impostazioni checkpoint
SAVE_EVERY_N_ITEMS = 5
MAX_ITEMS_TO_TEST = 3
NUM_EXAMPLES = 1

# percorsi
IMAGES_BASE_DIR = "/media/niki/Niki My Passport/scuola/universit/magistrale/tesi/dataset/Nutrition5k/nutrition5k_dataset"
TEST_DATASET_PATH = "datasets/Nutrition5k/data/ground truth/test_dataset.json"
OUTPUT_PREDICTIONS_PATH = "data/prediction/test_predictions_nutrition5k_twostep.json"


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
        "Look carefully at the image of this dish and compare it with the following preliminary extraction. "
        "Use the image to visually anchor the actual portion sizes, then calculate the exact macronutrients for the dish.\n\n"
        "Preliminary Food Details:\n"
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
        "You are an expert nutritionist and food vision analyst.\n"
        "Your task is to perform a precise nutritional breakdown by visually analyzing the image of the dish alongside a provided list of ingredients.\n\n"
        "CORE DIRECTIVES FOR STEP 2:\n"
        "1. LOOK AT THE IMAGE to assess the actual volume, density, and portion size of each item.\n"
        "2. Use the provided ingredient list and preliminary weights as a baseline, but you MUST visually validate and adjust the mass based on the real image.\n"
        "3. Calculate exact macronutrients (calories, fat, carbohydrates, protein) based on your combined visual and textual analysis.\n"
        "You MUST respond ONLY with a valid JSON."
    )

    # --- 2. CONFIGURAZIONE PROVIDER ---
    step1_sys_prompt = base_step1_instruction
    step2_sys_prompt = base_step2_instruction

    global OUTPUT_PREDICTIONS_PATH

    if PROVIDER == "gemma":
        model = os.environ.get("OLLAMA_GEMMA_MODEL", "gemma4:e4b")
        client = GemmaClient(model_name=model)
        step1_sys_prompt = "<|think|>\n" + base_step1_instruction
        step2_sys_prompt = "<|think|>\n" + base_step2_instruction
        OUTPUT_PREDICTIONS_PATH = "data/prediction/test_predictions_nutrition5k_twostep_gemma.json"

    elif PROVIDER == "qwen":
        model = os.environ.get("OLLAMA_QWEN_MODEL", "qwen3.5:4b")
        client = QwenClient(model_name=model)
        OUTPUT_PREDICTIONS_PATH = "data/prediction/test_predictions_nutrition5k_twostep_qwen.json"

    elif PROVIDER == "minicpm":
        model = os.environ.get("OLLAMA_MINICPM_MODEL", "openbmb/minicpm-v4.5:8b")
        client = MiniCPMClient(model_name=model, thinking_mode="deep")
        OUTPUT_PREDICTIONS_PATH = "data/prediction/test_predictions_nutrition5k_twostep_minicpm.json"

    else:
        raise ValueError(f"Provider '{PROVIDER}' non supportato.")

    print(f"[*] Modello selezionato: {model}")
    print(f"[*] File di output: {OUTPUT_PREDICTIONS_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS_PATH), exist_ok=True)

    # --- 3. FEW-SHOT EXAMPLES DIVISI PER STEP ---
    example_step1_output_1 = FoodAnalysisStep1(
        visible_ingredients=[
            BaseIngredient(
                name="arugula",
                weight_grams=21.13
            ),
            BaseIngredient(
                name="quinoa",
                weight_grams=211.29
            ),
            BaseIngredient(
                name="white rice",
                weight_grams=190.0
            ),
            BaseIngredient(
                name="parsley",
                weight_grams=4.23
            )
        ],
        hidden_ingredients=[
            BaseIngredient(
                name="olive oil",
                weight_grams=1.06
            ),
            BaseIngredient(
                name="salt",
                weight_grams=2.38
            ),
            BaseIngredient(
                name="garlic",
                weight_grams=5.28
            ),
            BaseIngredient(
                name="shallots",
                weight_grams=1.06
            ),
            BaseIngredient(
                name="thyme",
                weight_grams=1.06
            ),
            BaseIngredient(
                name="lemon juice",
                weight_grams=0.53
            )
        ],
        allergens=["None"]
    ).model_dump_json()

    example_step2_output_1 = NutritionExtraction(
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
        total_macronutrients=Macronutrients(calories=525.97, fat=5.87, carbohydrates=100.68, protein=15.52)
    ).model_dump_json()

    example_step1_output_2 = FoodAnalysisStep1(
        visible_ingredients=[
            BaseIngredient(name="quinoa", weight_grams=37.0),
            BaseIngredient(name="berries", weight_grams=147.0),
            BaseIngredient(name="bacon", weight_grams=38.0)
        ],
        hidden_ingredients=[],
        allergens=["None"]
    ).model_dump_json()

    example_step2_output_2 = NutritionExtraction(
        visible_ingredients=[
            IngredientWithNutrition(name="quinoa", weight_grams=37.0,
                                    macronutrients=Macronutrients(calories=44.4,
                                                                  fat=0.7,
                                                                  carbohydrates=7.77,
                                                                  protein=1.63)),
            IngredientWithNutrition(name="berries", weight_grams=147.0,
                                    macronutrients=Macronutrients(calories=83.79,
                                                                  fat=0.44,
                                                                  carbohydrates=20.58,
                                                                  protein=1.03)),
            IngredientWithNutrition(name="bacon", weight_grams=38.0,
                                    macronutrients=Macronutrients(calories=205.58,
                                                                  fat=15.96,
                                                                  carbohydrates=0.53,
                                                                  protein=14.06))
        ],
        hidden_ingredients=[],
        allergens=["None"],
        total_macronutrients=Macronutrients(calories=333.77, fat=17.1, carbohydrates=28.88, protein=16.72)
    ).model_dump_json()

    example_step1_output_3 = FoodAnalysisStep1(
        visible_ingredients=[
            BaseIngredient(name="tortillia chips", weight_grams=1.21),

            BaseIngredient(name="cheese", weight_grams=1.82),

            BaseIngredient(name="cheese pizza", weight_grams=54.00),

            BaseIngredient(name="squash", weight_grams=34.33),

            BaseIngredient(name="zucchini", weight_grams=34.33),

            BaseIngredient(name="carrot", weight_grams=29.58),

            BaseIngredient(name="broccoli", weight_grams=34.33),

            BaseIngredient(name="cherry tomatoes", weight_grams=1.82),

            BaseIngredient(name="corn", weight_grams=3.03),

            BaseIngredient(name="spinach (raw)", weight_grams=15.14),
        ],
        hidden_ingredients=[
            BaseIngredient(name="pepper", weight_grams=0.02),

            BaseIngredient(name="black beans", weight_grams=6.05),

            BaseIngredient(name="olive oil", weight_grams=0.30),

            BaseIngredient(name="salt", weight_grams=0.15),

            BaseIngredient(name="sour cream", weight_grams=1.21),

            BaseIngredient(name="chili", weight_grams=0.15),

            BaseIngredient(name="tomatoes", weight_grams=1.21),

            BaseIngredient(name="cherry tomatoes", weight_grams=1.82),

            BaseIngredient(name="onions", weight_grams=2.42),

            BaseIngredient(name="garlic", weight_grams=0.30),

            BaseIngredient(name="lime", weight_grams=0.61),
        ],
        allergens=["dairy"]
    ).model_dump_json()

    example_step2_output_3 = NutritionExtraction(
        visible_ingredients=[
            IngredientWithNutrition(name="tortillia chips", weight_grams=1.21,
                                    macronutrients=Macronutrients(calories=5.62,
                                                                  fat=3.41,
                                                                  carbohydrates=0.89,
                                                                  protein=0.10)),

            IngredientWithNutrition(name="cheese", weight_grams=1.82,
                                    macronutrients=Macronutrients(calories=7.30,
                                                                  fat=0.60,
                                                                  carbohydrates=0.02,
                                                                  protein=0.45)),

            IngredientWithNutrition(name="cheese pizza", weight_grams=54.00,
                                    macronutrients=Macronutrients(calories=143.59,
                                                                  fat=5.08,
                                                                  carbohydrates=18.09,
                                                                  protein=6.05)),

            IngredientWithNutrition(name="squash", weight_grams=34.33,
                                    macronutrients=Macronutrients(calories=7.04,
                                                                  fat=0.10,
                                                                  carbohydrates=1.17,
                                                                  protein=0.31)),

            IngredientWithNutrition(name="zucchini", weight_grams=34.33,
                                    macronutrients=Macronutrients(calories=5.15,
                                                                  fat=0.14,
                                                                  carbohydrates=0.93,
                                                                  protein=0.38)),

            IngredientWithNutrition(name="carrot", weight_grams=29.58,
                                    macronutrients=Macronutrients(calories=12.13,
                                                                  fat=0.06,
                                                                  carbohydrates=2.96,
                                                                  protein=0.27)),

            IngredientWithNutrition(name="broccoli", weight_grams=34.33,
                                    macronutrients=Macronutrients(calories=12.02,
                                                                  fat=0.14,
                                                                  carbohydrates=2.40,
                                                                  protein=0.82)),

            IngredientWithNutrition(name="cherry tomatoes", weight_grams=1.82,
                                    macronutrients=Macronutrients(calories=0.33,
                                                                  fat=0.0,
                                                                  carbohydrates=0.07,
                                                                  protein=0.02)),

            IngredientWithNutrition(name="corn", weight_grams=3.03,
                                    macronutrients=Macronutrients(calories=3.90,
                                                                  fat=0.5,
                                                                  carbohydrates=0.64,
                                                                  protein=0.10)),

            IngredientWithNutrition(name="spinach (raw)", weight_grams=15.14,
                                    macronutrients=Macronutrients(calories=3.48,
                                                                  fat=0.06,
                                                                  carbohydrates=0.55,
                                                                  protein=0.44)),
        ],
        hidden_ingredients=[
            IngredientWithNutrition(name="pepper", weight_grams=0.02,
                                    macronutrients=Macronutrients(calories=0.00,
                                                                  fat=0.00,
                                                                  carbohydrates=0.00,
                                                                  protein=0.00)),

            IngredientWithNutrition(name="black beans", weight_grams=6.05,
                                    macronutrients=Macronutrients(calories=7.99,
                                                                  fat=0.03,
                                                                  carbohydrates=1.44,
                                                                  protein=0.54)),

            IngredientWithNutrition(name="olive oil", weight_grams=0.30,
                                    macronutrients=Macronutrients(calories=2.61,
                                                                  fat=0.30,
                                                                  carbohydrates=0.0,
                                                                  protein=0.0)),

            IngredientWithNutrition(name="salt", weight_grams=0.15,
                                    macronutrients=Macronutrients(calories=0.0,
                                                                  fat=0.0,
                                                                  carbohydrates=0.0,
                                                                  protein=0.0)),

            IngredientWithNutrition(name="sour cream", weight_grams=1.21,
                                    macronutrients=Macronutrients(calories=2.40,
                                                                  fat=0.23,
                                                                  carbohydrates=0.06,
                                                                  protein=0.03)),

            IngredientWithNutrition(name="chili", weight_grams=0.15,
                                    macronutrients=Macronutrients(calories=0.06,
                                                                  fat=0.00,
                                                                  carbohydrates=0.13,
                                                                  protein=0.00)),

            IngredientWithNutrition(name="tomatoes", weight_grams=1.21,
                                    macronutrients=Macronutrients(calories=0.41,
                                                                  fat=0.0,
                                                                  carbohydrates=0.08,
                                                                  protein=0.01)),

            IngredientWithNutrition(name="cherry tomatoes", weight_grams=1.82,
                                    macronutrients=Macronutrients(calories=0.33,
                                                                  fat=0.0,
                                                                  carbohydrates=0.07,
                                                                  protein=0.02)),

            IngredientWithNutrition(name="onions", weight_grams=2.42,
                                    macronutrients=Macronutrients(calories=0.97,
                                                                  fat=0.0,
                                                                  carbohydrates=0.22,
                                                                  protein=0.03)),

            IngredientWithNutrition(name="garlic", weight_grams=0.30,
                                    macronutrients=Macronutrients(calories=0.45,
                                                                  fat=0.0,
                                                                  carbohydrates=0.1,
                                                                  protein=0.02)),

            IngredientWithNutrition(name="lime", weight_grams=0.61,
                                    macronutrients=Macronutrients(calories=0.18,
                                                                  fat=0.0,
                                                                  carbohydrates=0.06,
                                                                  protein=0.0)),
        ],
        allergens=["dairy"],
        total_macronutrients=Macronutrients(calories=214.64, fat=6.98, carbohydrates=29.68, protein=9.57)
    ).model_dump_json()

    example_step1_output_4 = FoodAnalysisStep1(
        visible_ingredients=[
            BaseIngredient(name="scrambled eggs", weight_grams=182.0),
            BaseIngredient(name="bacon", weight_grams=33.0)
        ],
        hidden_ingredients=[],
        allergens=["egg"]
    ).model_dump_json()

    example_step2_output_4 = NutritionExtraction(
        visible_ingredients=[
            IngredientWithNutrition(name="scrambled eggs", weight_grams=182.0,
                                    macronutrients=Macronutrients(calories=269.36,
                                                                  fat=20.02,
                                                                  carbohydrates=2.91,
                                                                  protein=18.2)),
            IngredientWithNutrition(name="bacon", weight_grams=33.0,
                                    macronutrients=Macronutrients(calories=178.53,
                                                                  fat=13.86,
                                                                  carbohydrates=0.46,
                                                                  protein=12.21))
        ],
        hidden_ingredients=[],
        allergens=["egg"],
        total_macronutrients=Macronutrients(calories=447.89, fat=33.88, carbohydrates=3.37, protein=30.41)
    ).model_dump_json()

    example_step1_output_5 = FoodAnalysisStep1(
        visible_ingredients=[
            BaseIngredient(name="mustard", weight_grams=0.18),

            BaseIngredient(name="spinach (raw)", weight_grams=17.72),

            BaseIngredient(name="radishes", weight_grams=2.84),

        ],
        hidden_ingredients=[
            BaseIngredient(name="salt", weight_grams=0.09),

            BaseIngredient(name="garlic", weight_grams=0.09),

            BaseIngredient(name="olive oil", weight_grams=2.84),

            BaseIngredient(name="blueberries", weight_grams=2.84),

            BaseIngredient(name="vinegar", weight_grams=1.42),
        ],
        allergens=["None"]
    ).model_dump_json()

    example_step2_output_5 = NutritionExtraction(
        visible_ingredients=[
            IngredientWithNutrition(name="mustard", weight_grams=0.18,
                                    macronutrients=Macronutrients(calories=0.11,
                                                                  fat=0.0,
                                                                  carbohydrates=0.01,
                                                                  protein=0.0)),

            IngredientWithNutrition(name="spinach (raw)", weight_grams=17.72,
                                    macronutrients=Macronutrients(calories=4.08,
                                                                  fat=0.07,
                                                                  carbohydrates=0.64,
                                                                  protein=0.51)),

            IngredientWithNutrition(name="radishes", weight_grams=2.84,
                                    macronutrients=Macronutrients(calories=0.45,
                                                                  fat=0.0,
                                                                  carbohydrates=0.1,
                                                                  protein=0.02)),

        ],
        hidden_ingredients=[
            IngredientWithNutrition(name="salt", weight_grams=0.09,
                                    macronutrients=Macronutrients(calories=0.0,
                                                                  fat=0.0,
                                                                  carbohydrates=0.0,
                                                                  protein=0.0)),

            IngredientWithNutrition(name="garlic", weight_grams=0.09,
                                    macronutrients=Macronutrients(calories=0.13,
                                                                  fat=0.0,
                                                                  carbohydrates=0.03,
                                                                  protein=0.0)),

            IngredientWithNutrition(name="olive oil", weight_grams=2.84,
                                    macronutrients=Macronutrients(calories=25.07,
                                                                  fat=2.84,
                                                                  carbohydrates=0.0,
                                                                  protein=0.0)),

            IngredientWithNutrition(name="blueberries", weight_grams=2.84,
                                    macronutrients=Macronutrients(calories=1.62,
                                                                  fat=0.0,
                                                                  carbohydrates=0.41,
                                                                  protein=0.02)),

            IngredientWithNutrition(name="vinegar", weight_grams=1.42,
                                    macronutrients=Macronutrients(calories=0.0,
                                                                  fat=0.0,
                                                                  carbohydrates=0.0,
                                                                  protein=0.0)),
        ],
        allergens=["None"],
        total_macronutrients=Macronutrients(calories=31.45, fat=2.92, carbohydrates=1.18, protein=0.57)
    ).model_dump_json()

    example_step1_output_6 = FoodAnalysisStep1(
        visible_ingredients=[
            BaseIngredient(name="sweet potato", weight_grams=118.05),

            BaseIngredient(name="watermelon", weight_grams=84.0),
            BaseIngredient(name="berries", weight_grams=93.0)
        ],
        hidden_ingredients=[
            BaseIngredient(name="olive oil", weight_grams=2.95),
        ],
        allergens=["None"]
    ).model_dump_json()

    example_step2_output_6 = NutritionExtraction(
        visible_ingredients=[
            IngredientWithNutrition(name="sweet potato", weight_grams=118.05,
                                    macronutrients=Macronutrients(calories=89.72,
                                                                  fat=0.0,
                                                                  carbohydrates=21.25,
                                                                  protein=1.18)),

            IngredientWithNutrition(name="watermelon", weight_grams=84.0,
                                    macronutrients=Macronutrients(calories=25.2,
                                                                  fat=0.17,
                                                                  carbohydrates=6.38,
                                                                  protein=0.5)),
            IngredientWithNutrition(name="berries", weight_grams=93.0,
                                    macronutrients=Macronutrients(calories=53.01,
                                                                  fat=0.28,
                                                                  carbohydrates=13.02,
                                                                  protein=0.65))
        ],
        hidden_ingredients=[
            IngredientWithNutrition(name="olive oil", weight_grams=2.95,
                                    macronutrients=Macronutrients(calories=26.09,
                                                                  fat=2.95,
                                                                  carbohydrates=0.0,
                                                                  protein=0.0)),
        ],
        allergens=["None"],
        total_macronutrients=Macronutrients(calories=194.02, fat=3.4, carbohydrates=40.65, protein=2.34)
    ).model_dump_json()

    example_step1_output_7 = FoodAnalysisStep1(
        visible_ingredients=[
            BaseIngredient(name="apple", weight_grams=6.37),

            BaseIngredient(name="white rice", weight_grams=13.66),

            BaseIngredient(name="squash", weight_grams=49.32),

            BaseIngredient(name="mustard greens", weight_grams=31.83),

            BaseIngredient(name="kale", weight_grams=3.41),

            BaseIngredient(name="steak", weight_grams=102.42),

            BaseIngredient(name="arugula", weight_grams=6.37),
            BaseIngredient(name="cilantro", weight_grams=0.34),
            BaseIngredient(name="figs", weight_grams=6.37),

            BaseIngredient(name="pineapple", weight_grams=3.41),
        ],
        hidden_ingredients=[
            BaseIngredient(name="vinegar", weight_grams=0.68),

            BaseIngredient(name="lemon juice", weight_grams=3.23),

            BaseIngredient(name="garlic", weight_grams=0.68),

            BaseIngredient(name="salt", weight_grams=0.51),

            BaseIngredient(name="olive oil", weight_grams=5.59),

            BaseIngredient(name="goat cheese", weight_grams=1.27),

            BaseIngredient(name="jalapenos", weight_grams=0.34),

            BaseIngredient(name="onions", weight_grams=8.19),
        ],
        allergens=["None"]
    ).model_dump_json()

    example_step2_output_7 = NutritionExtraction(
        visible_ingredients=[
            IngredientWithNutrition(name="apple", weight_grams=6.37,
                                    macronutrients=Macronutrients(calories=3.31,
                                                                  fat=0.01,
                                                                  carbohydrates=0.89,
                                                                  protein=0.02)),

            IngredientWithNutrition(name="white rice", weight_grams=13.66,
                                    macronutrients=Macronutrients(calories=17.75,
                                                                  fat=0.04,
                                                                  carbohydrates=3.82,
                                                                  protein=0.37)),

            IngredientWithNutrition(name="squash", weight_grams=49.32,
                                    macronutrients=Macronutrients(calories=10.11,
                                                                  fat=0.15,
                                                                  carbohydrates=1.68,
                                                                  protein=0.44)),

            IngredientWithNutrition(name="mustard greens", weight_grams=31.83,
                                    macronutrients=Macronutrients(calories=8.59,
                                                                  fat=0.13,
                                                                  carbohydrates=1.5,
                                                                  protein=0.92)),

            IngredientWithNutrition(name="kale", weight_grams=3.41,
                                    macronutrients=Macronutrients(calories=1.67,
                                                                  fat=0.02,
                                                                  carbohydrates=0.31,
                                                                  protein=0.11)),

            IngredientWithNutrition(name="steak", weight_grams=102.42,
                                    macronutrients=Macronutrients(calories=277.56,
                                                                  fat=19.46,
                                                                  carbohydrates=0.0,
                                                                  protein=25.61)),

            IngredientWithNutrition(name="arugula", weight_grams=6.37,
                                    macronutrients=Macronutrients(calories=1.59,
                                                                  fat=0.04,
                                                                  carbohydrates=0.24,
                                                                  protein=0.17)),
            IngredientWithNutrition(name="cilantro", weight_grams=0.34,
                                    macronutrients=Macronutrients(calories=0.08,
                                                                  fat=0.0,
                                                                  carbohydrates=0.01,
                                                                  protein=0.01)),
            IngredientWithNutrition(name="figs", weight_grams=6.37,
                                    macronutrients=Macronutrients(calories=17.51,
                                                                  fat=0.0,
                                                                  carbohydrates=4.14,
                                                                  protein=0.16)),

            IngredientWithNutrition(name="pineapple", weight_grams=3.41,
                                    macronutrients=Macronutrients(calories=1.71,
                                                                  fat=0.0,
                                                                  carbohydrates=0.44,
                                                                  protein=0.02)),
        ],
        hidden_ingredients=[
            IngredientWithNutrition(name="vinegar", weight_grams=0.68,
                                    macronutrients=Macronutrients(calories=0.0,
                                                                  fat=0.0,
                                                                  carbohydrates=0.0,
                                                                  protein=0.0)),

            IngredientWithNutrition(name="lemon juice", weight_grams=3.23,
                                    macronutrients=Macronutrients(calories=0.55,
                                                                  fat=0.0,
                                                                  carbohydrates=0.18,
                                                                  protein=0.01)),

            IngredientWithNutrition(name="garlic", weight_grams=0.68,
                                    macronutrients=Macronutrients(calories=1.02,
                                                                  fat=0.0,
                                                                  carbohydrates=0.23,
                                                                  protein=0.04)),

            IngredientWithNutrition(name="salt", weight_grams=0.51,
                                    macronutrients=Macronutrients(calories=0.0,
                                                                  fat=0.0,
                                                                  carbohydrates=0.0,
                                                                  protein=0.0)),

            IngredientWithNutrition(name="olive oil", weight_grams=5.59,
                                    macronutrients=Macronutrients(calories=49.38, fat=5.59,
                                                                  carbohydrates=0.0,
                                                                  protein=0.0)),

            IngredientWithNutrition(name="goat cheese", weight_grams=1.27,
                                    macronutrients=Macronutrients(calories=3.36,
                                                                  fat=0.27,
                                                                  carbohydrates=0.0,
                                                                  protein=0.24)),

            IngredientWithNutrition(name="jalapenos", weight_grams=0.34,
                                    macronutrients=Macronutrients(calories=0.1,
                                                                  fat=0.0,
                                                                  carbohydrates=0.02,
                                                                  protein=0.0)),

            IngredientWithNutrition(name="onions", weight_grams=8.19,
                                    macronutrients=Macronutrients(calories=3.28,
                                                                  fat=0.01,
                                                                  carbohydrates=0.74,
                                                                  protein=0.09)),
        ],
        allergens=["None"],
        total_macronutrients=Macronutrients(calories=397.57, fat=25.73, carbohydrates=14.19, protein=28.21)
    ).model_dump_json()

    example_step1_output_8 = FoodAnalysisStep1(
        visible_ingredients=[
            BaseIngredient(name="cherry tomatoes", weight_grams=10.7),

            BaseIngredient(name="mustard greens", weight_grams=21.4),

            BaseIngredient(name="asparagus", weight_grams=34.0),

            BaseIngredient(name="jalapegnos", weight_grams=1.07),

            BaseIngredient(name="mangos", weight_grams=10.70),


        ],
        hidden_ingredients=[

            BaseIngredient(name="tomatoes", weight_grams=10.7),

            BaseIngredient(name="olive oil", weight_grams=1.07),

            BaseIngredient(name="salt", weight_grams=0.13),

            BaseIngredient(name="lemon juice", weight_grams=1.61),

            BaseIngredient(name="lime", weight_grams=1.60),

            BaseIngredient(name="cilantro", weight_grams=1.07),

            BaseIngredient(name="onions", weight_grams=2.14),

            BaseIngredient(name="cucumbers", weight_grams=10.70),

            BaseIngredient(name="onions", weight_grams=2.14),

            BaseIngredient(name="arugula", weight_grams=32.10),


        ],
        allergens=["None"]
    ).model_dump_json()

    example_step2_output_8 = NutritionExtraction(
        visible_ingredients=[
            IngredientWithNutrition(name="cherry tomatoes", weight_grams=10.7,
                                    macronutrients=Macronutrients(calories=1.93,
                                                                  fat=0.02,
                                                                  carbohydrates=0.42,
                                                                  protein=0.10)),

            IngredientWithNutrition(name="mustard greens", weight_grams=21.4,
                                    macronutrients=Macronutrients(calories=5.78,
                                                                  fat=0.09,
                                                                  carbohydrates=1.01,
                                                                  protein=0.62)),

            IngredientWithNutrition(name="asparagus", weight_grams=34.0,
                                    macronutrients=Macronutrients(calories=6.94,
                                                                  fat=0.03,
                                                                  carbohydrates=1.33,
                                                                  protein=0.75)),

            IngredientWithNutrition(name="jalapegnos", weight_grams=1.07,
                                    macronutrients=Macronutrients(calories=0.31,
                                                                  fat=0.0,
                                                                  carbohydrates=0.07,
                                                                  protein=0.01)),

            IngredientWithNutrition(name="mangos", weight_grams=10.70,
                                    macronutrients=Macronutrients(calories=6.48,
                                                                  fat=0.04,
                                                                  carbohydrates=1.60,
                                                                  protein=0.09)),


        ],
        hidden_ingredients=[

            IngredientWithNutrition(name="tomatoes", weight_grams=10.7,
                                    macronutrients=Macronutrients(calories=3.64,
                                                                  fat=0.07,
                                                                  carbohydrates=0.73,
                                                                  protein=0.07)),

            IngredientWithNutrition(name="olive oil", weight_grams=1.07,
                                    macronutrients=Macronutrients(calories=9.46,
                                                                  fat=1.07,
                                                                  carbohydrates=0.0,
                                                                  protein=0.0)),

            IngredientWithNutrition(name="salt", weight_grams=0.13,
                                    macronutrients=Macronutrients(calories=0.0,
                                                                  fat=0.0,
                                                                  carbohydrates=0.0,
                                                                  protein=0.0)),

            IngredientWithNutrition(name="lemon juice", weight_grams=1.61,
                                    macronutrients=Macronutrients(calories=0.27,
                                                                  fat=0.0,
                                                                  carbohydrates=0.09,
                                                                  protein=0.01)),

            IngredientWithNutrition(name="lime", weight_grams=1.60,
                                    macronutrients=Macronutrients(calories=0.48,
                                                                  fat=0.0,
                                                                  carbohydrates=0.17,
                                                                  protein=0.01)),

            IngredientWithNutrition(name="cilantro", weight_grams=1.07,
                                    macronutrients=Macronutrients(calories=0.24,
                                                                  fat=0.01,
                                                                  carbohydrates=0.04,
                                                                  protein=0.02)),

            IngredientWithNutrition(name="onions", weight_grams=2.14,
                                    macronutrients=Macronutrients(calories=0.86,
                                                                  fat=0.80,
                                                                  carbohydrates=0.19,
                                                                  protein=0.02)),

            IngredientWithNutrition(name="cucumbers", weight_grams=10.70,
                                    macronutrients=Macronutrients(calories=1.07,
                                                                  fat=0.0,
                                                                  carbohydrates=0.21,
                                                                  protein=0.11)),

            IngredientWithNutrition(name="onions", weight_grams=2.14,
                                    macronutrients=Macronutrients(calories=0.86,
                                                                  fat=0.0,
                                                                  carbohydrates=0.19,
                                                                  protein=0.02)),

            IngredientWithNutrition(name="arugula", weight_grams=32.10,
                                    macronutrients=Macronutrients(calories=8.03,
                                                                  fat=0.22,
                                                                  carbohydrates=1.19,
                                                                  protein=0.83)),


        ],
        allergens=["None"],
        total_macronutrients=Macronutrients(calories=45.48, fat=1.57, carbohydrates=7.04, protein=2.64)
    ).model_dump_json()

    example_step1_output_9 = FoodAnalysisStep1(
        visible_ingredients=[
            BaseIngredient(name="scrambled eggs", weight_grams=68.0),
            BaseIngredient(name="yam", weight_grams=50.0),
            BaseIngredient(name="grapes", weight_grams=30.0),
            BaseIngredient(name="egg whites", weight_grams=67.0),
            BaseIngredient(name="sweet potato", weight_grams=50.0)
        ],
        hidden_ingredients=[
            BaseIngredient(name="olive oil", weight_grams=6.0)
        ],
        allergens=["egg"]
    ).model_dump_json()

    example_step2_output_9 = NutritionExtraction(
        visible_ingredients=[
            IngredientWithNutrition(name="scrambled eggs", weight_grams=68.0,
                                    macronutrients=Macronutrients(calories=100.64,
                                                                  fat=7.48,
                                                                  carbohydrates=1.09,
                                                                  protein=6.8)),
            IngredientWithNutrition(name="yam", weight_grams=50.0,
                                    macronutrients=Macronutrients(calories=62.05,
                                                                  fat=0.1,
                                                                  carbohydrates=14.7,
                                                                  protein=0.8)),
            IngredientWithNutrition(name="grapes", weight_grams=30.0,
                                    macronutrients=Macronutrients(calories=20.7,
                                                                  fat=0.06,
                                                                  carbohydrates=5.4,
                                                                  protein=0.21)),
            IngredientWithNutrition(name="egg whites", weight_grams=67.0,
                                    macronutrients=Macronutrients(calories=34.84,
                                                                  fat=0.13,
                                                                  carbohydrates=0.47,
                                                                  protein=6.7)),
            IngredientWithNutrition(name="sweet potato", weight_grams=50.0,
                                    macronutrients=Macronutrients(calories=38.0,
                                                                  fat=0.0,
                                                                  carbohydrates=9.0,
                                                                  protein=0.5))
        ],
        hidden_ingredients=[
            IngredientWithNutrition(name="olive oil", weight_grams=6.0,
                                    macronutrients=Macronutrients(calories=53.04,
                                                                  fat=6.0,
                                                                  carbohydrates=0.0,
                                                                  protein=0.0))
        ],
        allergens=["egg"],
        total_macronutrients=Macronutrients(calories=309.27, fat=13.77, carbohydrates=30.66, protein=15.01)
    ).model_dump_json()

    example_step1_output_10 = FoodAnalysisStep1(
        visible_ingredients=[
            BaseIngredient(name="eggplant", weight_grams=57.84),

            BaseIngredient(name="roasted potatoes", weight_grams=60.0),

            BaseIngredient(name="cauliflower", weight_grams=64.0)
        ],
        hidden_ingredients=[
            BaseIngredient(name="salt", weight_grams=0.58),
            BaseIngredient(name="olive oil", weight_grams=0.58)
        ],
        allergens=["egg"]
    ).model_dump_json()

    example_step2_output_10 = NutritionExtraction(
        visible_ingredients=[
            IngredientWithNutrition(name="eggplant", weight_grams=57.84,
                                    macronutrients=Macronutrients(calories=14.35,
                                                                  fat=0.12,
                                                                  carbohydrates=3.41,
                                                                  protein=0.58)),

            IngredientWithNutrition(name="roasted potatoes", weight_grams=60.0,
                                    macronutrients=Macronutrients(calories=84.6,
                                                                  fat=4.08,
                                                                  carbohydrates=10.8,
                                                                  protein=1.26)),

            IngredientWithNutrition(name="cauliflower", weight_grams=64.0,
                                    macronutrients=Macronutrients(calories=16.0,
                                                                  fat=0.19,
                                                                  carbohydrates=3.2,
                                                                  protein=1.15))
        ],
        hidden_ingredients=[
            IngredientWithNutrition(name="salt", weight_grams=0.58,
                                    macronutrients=Macronutrients(calories=0.0,
                                                                  fat=0.0,
                                                                  carbohydrates=0.0,
                                                                  protein=0.0)),
            IngredientWithNutrition(name="olive oil", weight_grams=0.58,
                                    macronutrients=Macronutrients(calories=5.11,
                                                                  fat=0.58,
                                                                  carbohydrates=0.0,
                                                                  protein=0.0))
        ],
        allergens=["egg"],
        total_macronutrients=Macronutrients(calories=120.06, fat=4.97, carbohydrates=17.41, protein=2.99)
    ).model_dump_json()

    all_few_shot_examples_step1 = [
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1562012076/rgb.png")), "output": example_step1_output_1},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1560442279/rgb.png")), "output": example_step1_output_2},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1563395449/rgb.png")), "output": example_step1_output_3},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1562086740/rgb.png")), "output": example_step1_output_4},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1561491121/rgb.png")), "output": example_step1_output_5},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1564590190/rgb.png")), "output": example_step1_output_6},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1562615128/rgb.png")), "output": example_step1_output_7},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1565640549/rgb.png")), "output": example_step1_output_8},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1563207364/rgb.png")), "output": example_step1_output_9},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1561575474/rgb.png")), "output": example_step1_output_10},
    ]

    all_few_shot_examples_step2 = [
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1562012076/rgb.png")), "output": example_step2_output_1},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1560442279/rgb.png")), "output": example_step2_output_2},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1563395449/rgb.png")), "output": example_step2_output_3},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1562086740/rgb.png")), "output": example_step2_output_4},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1561491121/rgb.png")), "output": example_step2_output_5},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1564590190/rgb.png")), "output": example_step2_output_6},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1562615128/rgb.png")), "output": example_step2_output_7},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1565640549/rgb.png")), "output": example_step2_output_8},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1563207364/rgb.png")), "output": example_step2_output_9},
        {"image_path": normalize_path(os.path.join(IMAGES_BASE_DIR, "imagery/realsense_overhead/dish_1561575474/rgb.png")), "output": example_step2_output_10},
    ]

    print(f"number of examples in prompt: {NUM_EXAMPLES}")
    examples_step1 = all_few_shot_examples_step1[:NUM_EXAMPLES]
    examples_step2 = all_few_shot_examples_step2[:NUM_EXAMPLES]

    # --- CARICAMENTO DATI ---
    with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if MAX_ITEMS_TO_TEST:
        test_data = test_data[:MAX_ITEMS_TO_TEST]
        print(f"[*] MAX_ITEMS_TO_TEST active: execution limited to {MAX_ITEMS_TO_TEST}")

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