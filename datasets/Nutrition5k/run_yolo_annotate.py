"""
run_yolo_annotate.py
====================
Runs YOLO on the specified dishes (as in ExtractFoodInformation.py)
and saves the annotated images with the detected labels.

Usage:
    python run_yolo_annotate.py
    python run_yolo_annotate.py --weights /alternative/path/yolov8s.pt
    python run_yolo_annotate.py --out data/yolo_annotated
"""

import os
import sys
import argparse
from collections import defaultdict
from settings import get_env, get_path  # centralized configuration (.env)

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Annotate dish images with YOLO.")
    p.add_argument("--weights", default=None,
                   help="Path to the YOLO .pt weights (default: yolo_weights/yolov8s.pt relative to the project root)")
    p.add_argument("--out", default="data/yolo_annotated",
                   help="Output folder for the annotated images (default: data/yolo_annotated)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Fixed paths (same as ExtractFoodInformation.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.dirname(os.path.dirname(SCRIPT_DIR))
IMAGES_BASE_DIR = get_path("NUTRITION5K_DATASET_DIR")

DISH_IDS = [
    "dish_1562008979",
    "dish_1565711527",
]

# ---------------------------------------------------------------------------
# Functions copied from ExtractFoodInformation.py (without the OllamaClient dependency)
# ---------------------------------------------------------------------------

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


_yolo_model = None

def run_yolo_on_image(full_image_path, weights_path, output_dir=None):
    """Same as ExtractFoodInformation.run_yolo_on_image, but receives
    the weights as an explicit parameter instead of reading the global variable."""
    import cv2
    from ultralytics import YOLO
    from ultralytics.utils.plotting import colors, Annotator
    global _yolo_model

    if _yolo_model is None:
        print(f"[INFO] Loading YOLO model: {weights_path}")
        _yolo_model = YOLO(weights_path)

    yolo_image_modified = None
    yolo_extracted_data = ""

    img = cv2.imread(full_image_path)
    if img is None:
        print(f"[Warning] Unable to read the image with cv2: {full_image_path}")
        return None, ""

    img_resized = cv2.resize(img, (640, 640))
    results = _yolo_model(img_resized)

    raw_boxes = []
    for box in results[0].boxes:
        class_id   = int(box.cls[0].item())
        class_name = _yolo_model.names[class_id]
        coords     = box.xyxy[0].tolist()
        conf       = float(box.conf[0].item()) if hasattr(box, 'conf') and box.conf is not None else 1.0
        raw_boxes.append({
            "class_name": class_name,
            "coords": coords,
            "conf": conf,
            "class_id": class_id,
            "weight": conf,
            "count": 1,
        })

    print(f"  Raw YOLO objects: {[b['class_name'] for b in raw_boxes]}")

    # --- NMS filtering for IoU > 0.5 between boxes of the same class ---
    IOU_THRESHOLD = 0.5
    filtered_boxes = []
    boxes_by_class = {}
    for box in raw_boxes:
        cname = box["class_name"]
        boxes_by_class.setdefault(cname, []).append(box)

    for cname, class_boxes in boxes_by_class.items():
        groups = []
        for box in class_boxes:
            matching_groups_indices = []
            for i, group in enumerate(groups):
                overlaps = any(
                    calculate_iou(box["coords"], g_box["coords"]) > IOU_THRESHOLD
                    for g_box in group
                )
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
                for k in range(4):
                    avg_coords[k] += c[k] * w
            avg_coords = [x / total_weight for x in avg_coords]
            filtered_boxes.append({
                "class_name": cname,
                "class_id":   group[0]["class_id"],
                "coords":     avg_coords,
                "conf":       total_weight / len(group),
                "weight":     total_weight,
                "count":      len(group),
            })

    filtered_boxes.sort(key=lambda x: x["conf"], reverse=True)

    # --- Image annotation ---
    annotator = Annotator(img_resized.copy(), line_width=2,
                          example=str(_yolo_model.names))
    ing_groups = defaultdict(list)

    for idx, box in enumerate(filtered_boxes, start=1):
        coords     = box["coords"]
        class_id   = box["class_id"]
        class_name = box["class_name"]
        annotator.box_label(coords, str(idx), color=colors(class_id, True))

        box_area  = (coords[2] - coords[0]) * (coords[3] - coords[1])
        rel_area  = (box_area / (640.0 * 640.0)) * 100.0
        ing_groups[class_name].append({"id": idx, "area": rel_area})

    for name, items in ing_groups.items():
        ids        = [str(item["id"]) for item in items]
        total_area = sum(item["area"] for item in items)
        if len(ids) == 1:
            id_str    = f"ID {ids[0]}"
            shape_str = "Compact shape"
        elif len(ids) == 2:
            id_str    = f"ID {ids[0]} & {ids[1]}"
            shape_str = "Scattered shape"
        else:
            id_str    = f"ID {', '.join(ids[:-1])} & {ids[-1]}"
            shape_str = "Scattered shape"
        yolo_extracted_data += (
            f"- {id_str} ({name}): {shape_str}. "
            f"Total area covered: {total_area:.1f}%.\n"
        )

    drawn_img = annotator.result()

    # --- Saving ---
    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "data", "prediction")
    os.makedirs(output_dir, exist_ok=True)

    dish_folder_name   = os.path.basename(os.path.dirname(full_image_path))
    safe_filename      = f"annotated_{dish_folder_name}.png"
    yolo_image_modified = os.path.join(output_dir, safe_filename)
    cv2.imwrite(yolo_image_modified, drawn_img)

    return yolo_image_modified, yolo_extracted_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.chdir(SCRIPT_DIR)
    args = parse_args()

    weights_path = args.weights or os.path.join(PROJECT_ROOT, "yolo_weights", "yolov8s.pt")
    output_dir   = os.path.join(SCRIPT_DIR, args.out)

    if not os.path.exists(weights_path):
        print(f"[ERROR] YOLO weights not found: {weights_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  YOLO annotation — weights: {os.path.basename(weights_path)}")
    print(f"  Output dir: {output_dir}")
    print(f"{'=' * 60}\n")

    for dish_id in DISH_IDS:
        rel_path  = f"imagery/realsense_overhead/{dish_id}/rgb.png"
        full_path = os.path.join(IMAGES_BASE_DIR, rel_path)

        print(f"{'─' * 60}")
        print(f"[DISH] {dish_id}")
        print(f"[PATH]   {full_path}")

        if not os.path.exists(full_path):
            print(f"[ERROR] Image not found: {full_path}")
            continue

        annotated_path, extracted_text = run_yolo_on_image(
            full_path, weights_path, output_dir=output_dir
        )

        if annotated_path:
            print(f"[OUTPUT] Annotated image saved in:\n         {annotated_path}")
        else:
            print(f"[ERROR] No annotated image produced.")

        print(f"\n[EXTRACTED LABELS]")
        if extracted_text.strip():
            for line in extracted_text.strip().splitlines():
                print(f"  {line}")
        else:
            print("  (no objects detected)")
        print()

    print(f"{'=' * 60}")
    print(f"  Done. Images in: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
