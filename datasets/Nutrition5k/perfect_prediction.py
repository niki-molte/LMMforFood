import json


def create_perfect_predictions(input_file="data/ground truth/test_dataset.json", output_file="data/prediction/perfect_predictions.json"):
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    perfect_predictions = []

    for item in dataset:
        pred_record = {
            "image_path": item["image_path"],
            "food_id": item["food_id"],
            "food_category": item.get("food_category", "Nutrition5k Plate"),

            "ground_truth_ingredients": item["ingredients"],

            "predicted_nutrition": {
                "visible_ingredients": item["ingredients"],
                "hidden_ingredients": [],
                "allergens": ["None"],
                "total_macronutrients": {
                    "calories": item["total_macronutrients"]["calories"],
                    "fat": item["total_macronutrients"]["fat"],
                    "carbohydrates": item["total_macronutrients"]["carbohydrates"],
                    "protein": item["total_macronutrients"]["protein"]
                }
            }
        }
        perfect_predictions.append(pred_record)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(perfect_predictions, f, indent=4)

    print(f"File '{output_file}' created successfully! It contains {len(perfect_predictions)} perfect predictions.")



if __name__ == "__main__":
    create_perfect_predictions()