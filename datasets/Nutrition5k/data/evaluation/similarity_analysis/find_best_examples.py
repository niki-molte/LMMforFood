import json
import os
import glob

folder = "/home/niki/PycharmProjects/LLMforFood/datasets/Nutrition5k/data/evaluation/similarity_analysis"
files = glob.glob(os.path.join(folder, "*.json"))

for file in sorted(files):
    with open(file, 'r') as f:
        data = json.load(f)
    
    best_dish = None
    best_score = -1
    best_info = None
    
    for dish_id, ingredients in data.items():
        if len(ingredients) < 5 or len(ingredients) > 10:
            continue
            
        total_passed = sum(1 for ing, info in ingredients.items() if info["threshold_passed"])
        avg_score = sum(info["max_similarity_score"] for ing, info in ingredients.items()) / len(ingredients)
        
        # We want all to pass and high average score
        if total_passed == len(ingredients):
            score = len(ingredients) + avg_score # Favor more ingredients
            if score > best_score:
                best_score = score
                best_dish = dish_id
                best_info = ingredients
                
    model_name = os.path.basename(file).replace("similarity_analysis_", "").replace(".json", "")
    print(f"### Model: {model_name}")
    print(f"**Best Dish ID:** `{best_dish}`")
    if best_info:
        print(f"**Ingredients:** ({len(best_info)} total, all matched)")
        for ing, info in best_info.items():
            print(f"- **{ing}** -> `{info['best_vocab_match']}` (Similarity: {info['max_similarity_score']:.2f})")
    print("")

