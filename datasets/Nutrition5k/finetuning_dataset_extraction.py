import json
import os
import sys
import random

# Add the project root to sys.path to import answer_schema
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from answer_schema.extraction import NutritionExtraction, IngredientWithNutrition, Macronutrients

def jsonl_finetuning_gemma(json_input_path, jsonl_output_path, jsonl_val_output_path):
    with open(json_input_path, "r") as f:
        data = json.load(f)

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

    few_shot_dishes = {
        "dish_1560442279",
        "dish_1561491121",
        "dish_1561575474",
        "dish_1562012076",
        "dish_1562086740",
        "dish_1562615128",
        "dish_1563207364",
        "dish_1563395449",
        "dish_1564590190",
        "dish_1565640549"
    }

    all_samples = []

    for entry in data:
        # Keep the exact relative path (e.g. imagery/realsense_overhead/dish_.../rgb.png)
        img_path = entry["image_path"]

        if any(dish_id in img_path for dish_id in few_shot_dishes):
            continue

        ingredients = entry.get("ingredients", [])
        if any("deprecated" in ing["name"].lower() for ing in ingredients): 
            continue

        visible_ingredients = []
        hidden_ingredients = []
        total_calories = 0.0
        total_fat = 0.0
        total_carbohydrates = 0.0
        total_protein = 0.0

        for ing in ingredients:
            m = ing["macronutrients"]
            
            macronutrients_obj = Macronutrients(
                calories=round(m["calories"], 2),
                fat=round(m["fat"], 2),
                carbohydrates=round(m["carbohydrates"], 2),
                protein=round(m["protein"], 2)
            )
            
            ing_obj = IngredientWithNutrition(
                name=ing["name"],
                weight_grams=round(ing["weight_grams"], 2),
                macronutrients=macronutrients_obj
            )

            if any(kw in ing["name"].lower() for kw in invisible_keywords):
                hidden_ingredients.append(ing_obj)
            else:
                visible_ingredients.append(ing_obj)

            total_calories += m["calories"]
            total_fat += m["fat"]
            total_carbohydrates += m["carbohydrates"]
            total_protein += m["protein"]

        if not visible_ingredients and not hidden_ingredients:
            continue

        total_macronutrients = Macronutrients(
            calories=round(total_calories, 2),
            fat=round(total_fat, 2),
            carbohydrates=round(total_carbohydrates, 2),
            protein=round(total_protein, 2)
        )

        nutrition_extraction = NutritionExtraction(
            visible_ingredients=visible_ingredients,
            hidden_ingredients=hidden_ingredients,
            allergens=["None"],
            total_macronutrients=total_macronutrients
        )

        sample = {
            "images": [img_path.replace("\\", "/")],  # Clean relative path
            "messages": [
                {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Analyze this dish. Provide the full breakdown of visible/hidden ingredients, "
                                         "estimated weights, nutritional values and allergens. Output ONLY the JSON."}
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": nutrition_extraction.model_dump_json()}]}
            ]
        }
        all_samples.append(sample)

    random.seed(42)
    random.shuffle(all_samples)

    val_size = int(len(all_samples) * 0.1)
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]

    with open(jsonl_output_path, "w", encoding="utf-8") as out_file:
        for s in train_samples:
            out_file.write(json.dumps(s) + "\n")

    with open(jsonl_val_output_path, "w", encoding="utf-8") as val_file:
        for s in val_samples:
            val_file.write(json.dumps(s) + "\n")

    print(f"Created {len(train_samples)} train examples in {jsonl_output_path}")
    print(f"Created {len(val_samples)} validation examples in {jsonl_val_output_path}")

# Local execution:
jsonl_finetuning_gemma(
    json_input_path="datasets/Nutrition5k/data/ground truth/train_dataset.json",
    jsonl_output_path="datasets/Nutrition5k/data/ground truth/finetuning_train_data.jsonl",
    jsonl_val_output_path="datasets/Nutrition5k/data/ground truth/finetuning_validation_data.jsonl"
)
