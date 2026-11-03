"""
inspect_dish.py
===============
Given a food_id, prints for each JSON file in data/prediction/five_example
the predicted ingredients (non-standardized and standardized) compared
with the ground truth.

Usage:
    python inspect_dish.py <food_id>            dish_1562008979     dish_1565711527
    python inspect_dish.py dish_1561662216
    python inspect_dish.py dish_1561662216 --threshold 0.6
    python inspect_dish.py dish_1561662216 --no-std   # skip standardization (faster)

Options:
    --threshold FLOAT   Cosine similarity threshold for standardization (default: 0.6)
    --no-std            Show only raw ingredients, without loading the NLP model
    --pred-dir PATH     Predictions directory (default: data/prediction/five_example)
    --vocab PATH        Vocabulary CSV (default: data/metadata/ingredients_metadata.csv)
"""

import os
import sys
import json
import argparse
import textwrap

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect the predictions of a dish in the five_example set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("food_id", help="Dish ID (e.g. dish_1561662216)")
    p.add_argument("--threshold", type=float, default=0.6,
                   help="Cosine similarity threshold for standardization (default: 0.6)")
    p.add_argument("--no-std", action="store_true",
                   help="Skip NLP standardization (faster, shows only raw)")
    p.add_argument("--pred-dir", default="data/prediction/five_example",
                   help="Directory with the prediction JSON files")
    p.add_argument("--vocab", default="data/metadata/ingredients_metadata.csv",
                   help="Ingredient vocabulary CSV")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Vocabulary + NLP model loading (only if needed)
# ---------------------------------------------------------------------------

def load_vocab(csv_path: str) -> list:
    import csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Vocabulary not found: {csv_path}")
    vocab = set()
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ing = row.get("ingr", "").strip().lower()
            if ing:
                vocab.add(ing)
    return list(vocab)


def standardize(names: list, nlp_model, vocabulary: list, vocab_embeddings, threshold: float) -> list:
    """Return the list of standardized ingredients (one by one)."""
    if not names:
        return []
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    embeddings = nlp_model.encode(names)
    sim_matrix = cosine_similarity(embeddings, vocab_embeddings)
    result = []
    for i, raw in enumerate(names):
        best_idx = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i][best_idx])
        if best_score >= threshold:
            result.append(vocabulary[best_idx])
        else:
            result.append(raw)   # left unchanged if below threshold
    return result


# ---------------------------------------------------------------------------
# Helper: parse ingredients from a JSON entry
# ---------------------------------------------------------------------------

def _get_name(ing: dict) -> str:
    """Extract the name from an ingredient dictionary (handles different formats)."""
    for key in ("name", "matched_ingredient", "original_prediction"):
        val = ing.get(key, "")
        if val:
            return str(val).strip().lower()
    return ""


def extract_predicted_names(entry: dict) -> list[str]:
    """Return the raw names of the predicted ingredients (visible + hidden).

    Handles the different formats:
    - Standard / YOLO:      predicted_nutrition → visible_ingredients + hidden_ingredients
    - Two-Step / Hybrid:    predicted_nutrition_step2 → visible_ingredients + hidden_ingredients
                            (or predicted_food_analysis_step1 as fallback)
    - Semantic Matching:    semantic_matching_validated_ingredients at root level (takes priority)
    """
    # Semantic Matching: takes precedence over everything
    if "semantic_matching_validated_ingredients" in entry:
        raw_list = entry["semantic_matching_validated_ingredients"]
        names = [_get_name(ing) for ing in raw_list if isinstance(ing, dict)]
        return [n for n in names if n]
    if "rag_validated_ingredients" in entry:
        raw_list = entry["rag_validated_ingredients"]
        names = [_get_name(ing) for ing in raw_list if isinstance(ing, dict)]
        return [n for n in names if n]

    # Look for the right key in order of preference
    pn = None
    for key in ("predicted_nutrition_step2", "predicted_nutrition",
                "predicted_food_analysis_step1"):
        candidate = entry.get(key)
        if isinstance(candidate, dict):
            pn = candidate
            break

    if pn is None:
        return []

    raw_list = pn.get("visible_ingredients", []) + pn.get("hidden_ingredients", [])
    names = [_get_name(ing) for ing in raw_list if isinstance(ing, dict)]
    return [n for n in names if n]


def extract_gt_names(entry: dict) -> list[str]:
    """Return the raw names of the ground truth ingredients."""
    gt = entry.get("ground_truth_ingredients", [])
    names = [_get_name(ing) for ing in gt if isinstance(ing, dict)]
    return [n for n in names if n]


# ---------------------------------------------------------------------------
# Helper: human-readable model name from the file name
# ---------------------------------------------------------------------------

def model_label(filename: str) -> str:
    """Convert the file name into a compact label for the table."""
    name = filename
    name = name.replace("test_predictions_nutrition5k_", "")
    name = name.replace("_tb.json", "").replace(".json", "")

    # Prompting technique
    if "v2hybrid_twostep" in name:
        technique = "Hybrid v2"
    elif "hybrid_twostep" in name:
        technique = "Hybrid v1"
    elif "twostep" in name:
        technique = "Two-Step"
    elif "yolo" in name:
        technique = "YOLO"
    else:
        technique = "Standard"

    # Hardware model
    hw = ""
    if "e2b" in name and "q4" in name:
        hw = "E2B-Q4"
    elif "e2b" in name and "q8" in name:
        hw = "E2B-Q8"
    elif "e4b" in name and "q4" in name:
        hw = "E4B-Q4"
    elif "e4b" in name and "q8" in name:
        hw = "E4B-Q8"

    return f"{technique} / {hw}" if hw else technique


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int) -> list[str]:
    """Split a string into lines of at most `width` characters."""
    if not text:
        return [""]
    return textwrap.wrap(text, width) or [""]


def print_table(food_id: str, gt_raw: list[str], rows: list[dict], show_std: bool):
    """
    rows: list of dicts with keys:
        label        str   model label
        pred_raw     list  non-standardized ingredients
        pred_std     list  standardized ingredients (may be None)
    """
    COL_MODEL  = 22
    COL_INGR   = 42   # raw / std ingredients column

    sep_thin  = "─"
    sep_thick = "═"

    def hline(char="─"):
        if show_std:
            print(f"┌{'─' * COL_MODEL}┬{'─' * COL_INGR}┬{'─' * COL_INGR}┐".replace("─", char))
        else:
            print(f"┌{'─' * COL_MODEL}┬{'─' * COL_INGR}┐".replace("─", char))

    def hline_mid(char="─"):
        if show_std:
            print(f"├{'─' * COL_MODEL}┼{'─' * COL_INGR}┼{'─' * COL_INGR}┤".replace("─", char))
        else:
            print(f"├{'─' * COL_MODEL}┼{'─' * COL_INGR}┤".replace("─", char))

    def hline_bot(char="─"):
        if show_std:
            print(f"└{'─' * COL_MODEL}┴{'─' * COL_INGR}┴{'─' * COL_INGR}┘".replace("─", char))
        else:
            print(f"└{'─' * COL_MODEL}┴{'─' * COL_INGR}┘".replace("─", char))

    def row_line(col1, col2, col3=None):
        c1 = f" {col1:<{COL_MODEL - 1}}"
        c2 = f" {col2:<{COL_INGR - 1}}"
        if show_std:
            c3 = f" {col3 or '':<{COL_INGR - 1}}"
            print(f"│{c1}│{c2}│{c3}│")
        else:
            print(f"│{c1}│{c2}│")

    def multiline_row(label: str, items_raw: list[str], items_std: list[str] | None = None):
        """Print a row that may span multiple lines for the ingredients."""
        raw_str  = ", ".join(items_raw)  if items_raw  else "(none)"
        std_str  = ", ".join(items_std)  if items_std  else "(none)"

        raw_lines = _wrap(raw_str,  COL_INGR - 2)
        std_lines = _wrap(std_str,  COL_INGR - 2) if show_std else []
        lbl_lines = _wrap(label,    COL_MODEL - 2)

        max_rows = max(len(raw_lines), len(std_lines), len(lbl_lines))

        for i in range(max_rows):
            l = lbl_lines[i] if i < len(lbl_lines) else ""
            r = raw_lines[i] if i < len(raw_lines) else ""
            s = std_lines[i] if i < len(std_lines) else ""
            row_line(l, r, s if show_std else None)

    # ---- Header ----
    title = f"  Dish: {food_id}  (5-example prompting)"
    total_w = COL_MODEL + COL_INGR + (COL_INGR + 1 if show_std else 0) + 3
    print()
    print("═" * total_w)
    print(title.center(total_w))
    print("═" * total_w)

    hline("═")
    row_line("Model / Technique", "Ingredients (Raw)", "Ingredients (Standardized)" if show_std else None)
    hline_mid("═")

    # ---- Ground truth ----
    multiline_row("★ GROUND TRUTH", gt_raw, gt_raw if show_std else None)
    hline_mid()

    # ---- Model predictions ----
    for i, r in enumerate(rows):
        multiline_row(r["label"], r["pred_raw"], r.get("pred_std"))
        if i < len(rows) - 1:
            hline_mid()

    hline_bot()

    # ---- TP/FP/FN summary ----
    print()
    gt_set = set(gt_raw)
    print(f"{'Model / Technique':<22}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'IoU':>6}"
          + (f"  {'TP_s':>4}  {'FP_s':>4}  {'FN_s':>4}  {'Prec_s':>7}  {'Rec_s':>7}  {'F1_s':>6}  {'IoU_s':>6}" if show_std else ""))
    print("-" * (90 + (65 if show_std else 0)))

    def metrics(pred_set, gt_set):
        intersection = len(pred_set & gt_set)
        union        = len(pred_set | gt_set)
        tp   = intersection
        fp   = len(pred_set - gt_set)
        fn   = len(gt_set - pred_set)
        prec = tp / len(pred_set) if pred_set else 0.0
        rec  = tp / len(gt_set)   if gt_set   else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        iou  = intersection / union           if union > 0         else 0.0
        return tp, fp, fn, prec, rec, f1, iou

    for r in rows:
        pred_raw_set = set(r["pred_raw"])
        tp, fp, fn, prec, rec, f1, iou = metrics(pred_raw_set, gt_set)
        line = f"{r['label']:<22}  {tp:>4}  {fp:>4}  {fn:>4}  {prec:>6.2%}  {rec:>6.2%}  {f1:>6.2%}  {iou:>6.2%}"
        if show_std and r.get("pred_std") is not None:
            std_set = set(r["pred_std"])
            tp2, fp2, fn2, prec2, rec2, f12, iou2 = metrics(std_set, gt_set)
            line += f"  {tp2:>4}  {fp2:>4}  {fn2:>4}  {prec2:>7.2%}  {rec2:>7.2%}  {f12:>6.2%}  {iou2:>6.2%}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Ensure the script works whether launched from the project root
    # or from its own directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    args = parse_args()
    food_id = args.food_id.strip()

    pred_dir  = args.pred_dir
    vocab_csv = args.vocab
    threshold = args.threshold
    do_std    = not args.no_std

    if not os.path.isdir(pred_dir):
        print(f"[ERROR] Predictions directory not found: {pred_dir}")
        sys.exit(1)

    # Collect all JSON files (excluding optuna files)
    json_files = sorted(
        f for f in os.listdir(pred_dir)
        if f.endswith(".json") and "optuna" not in f.lower()
    )
    if not json_files:
        print(f"[ERROR] No JSON file found in: {pred_dir}")
        sys.exit(1)

    # ---- Load NLP model + vocabulary if needed ----
    nlp_model = None
    vocabulary = None
    vocab_embeddings = None

    if do_std:
        print(f"[INFO] Loading vocabulary from {vocab_csv}...")
        vocabulary = load_vocab(vocab_csv)
        print(f"[INFO] Vocabulary: {len(vocabulary)} entries.")
        print("[INFO] Loading NLP model (all-MiniLM-L6-v2)...")
        from sentence_transformers import SentenceTransformer
        import numpy as np
        nlp_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[INFO] Computing vocabulary embeddings...")
        vocab_embeddings = nlp_model.encode(vocabulary)
        print("[INFO] NLP resources ready.\n")

    # ---- Look for the food_id in the JSON files ----
    gt_raw = None
    rows = []

    for fname in json_files:
        fpath = os.path.join(pred_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"[WARN] {fname}: unexpected format (not a list), skipped.")
            continue

        # Look for the entry with the requested food_id
        entry = next((e for e in data if e.get("food_id") == food_id), None)
        if entry is None:
            # Also try with image_path
            entry = next(
                (e for e in data if food_id in e.get("image_path", "")),
                None
            )
        if entry is None:
            print(f"[WARN] {fname}: food_id '{food_id}' not found.")
            continue

        # Ground truth (same for all files, taken from the first occurrence)
        if gt_raw is None:
            gt_raw = extract_gt_names(entry)

        pred_raw = extract_predicted_names(entry)

        pred_std = None
        if do_std and nlp_model is not None and pred_raw:
            pred_std = standardize(pred_raw, nlp_model, vocabulary, vocab_embeddings, threshold)

        rows.append({
            "label":    model_label(fname),
            "pred_raw": pred_raw,
            "pred_std": pred_std,
        })

    if not rows:
        print(f"\n[ERROR] food_id '{food_id}' was not found in any JSON file in '{pred_dir}'.")
        print("Hint: check the ID with --list (or inspect the files manually).")
        sys.exit(1)

    if gt_raw is None:
        gt_raw = []

    print_table(food_id, gt_raw, rows, show_std=do_std)


if __name__ == "__main__":
    main()
