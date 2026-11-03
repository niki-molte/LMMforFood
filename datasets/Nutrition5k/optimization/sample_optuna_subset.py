import json
import random
import re
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATASET_PATH = os.path.join(SCRIPT_DIR, "../data", "ground truth", "train_dataset.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "../data", "ground truth", "train_set_optuna.json")
CODE_PATH = os.path.join(SCRIPT_DIR, "../ExtractFoodInformation.py")

def main():
    # 1. Extract the IDs of the dishes used as few-shot examples from the source code
    with open(CODE_PATH, 'r', encoding='utf-8') as f:
        code_content = f.read()
    
    # Find all patterns like "dish_1562012076"
    excluded_dishes = set(re.findall(r'dish_\d+', code_content))
    print(f"[*] Found {len(excluded_dishes)} dishes excluded from ExtractFoodInformation.py:")
    for d in sorted(list(excluded_dishes)):
        print(f"    - {d}")

    # 2. Load the training dataset
    print(f"[*] Loading training dataset from: {TRAIN_DATASET_PATH}")
    with open(TRAIN_DATASET_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print(f"[*] Original train set size: {len(train_data)} items.")

    # 3. Random sampling with a fixed seed over the whole dataset
    random.seed(42)
    sample_size = 507
    
    if len(train_data) < sample_size:
        print(f"[!] Warning: the original data ({len(train_data)}) is smaller than the requested sample ({sample_size}).")
        sampled_data = list(train_data)
    else:
        sampled_data = random.sample(train_data, sample_size)

    # 4. Replace the dishes used as examples if they were sampled
    # Build a set of the sampled identifiers to exclude them from the pool
    sampled_ids = {id(item) for item in sampled_data}
    
    # Prepare a pool of valid items for replacements (not excluded and not already sampled)
    valid_pool = []
    for item in train_data:
        if id(item) in sampled_ids:
            continue
            
        img_path = item.get("image_path", "")
        match = re.search(r'dish_\d+', img_path)
        if match:
            dish_id = match.group(0)
            if dish_id not in excluded_dishes:
                valid_pool.append(item)
        else:
            valid_pool.append(item)
            
    # Replace the invalid items in sampled_data
    final_sampled_data = []
    replace_count = 0
    for item in sampled_data:
        img_path = item.get("image_path", "")
        match = re.search(r'dish_\d+', img_path)
        is_excluded = False
        if match:
            dish_id = match.group(0)
            if dish_id in excluded_dishes:
                is_excluded = True
                
        if is_excluded and valid_pool:
            replacement = random.choice(valid_pool)
            valid_pool.remove(replacement)
            final_sampled_data.append(replacement)
            replace_count += 1
        else:
            final_sampled_data.append(item)

    sampled_data = final_sampled_data
    
    print(f"[*] Replaced {replace_count} example dishes that had been sampled.")
    print(f"[*] Sampled {len(sampled_data)} items.")

    # 5. Save the new dataset
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=4, ensure_ascii=False)
        
    print(f"[*] Sampled subset saved successfully to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
