"""
data_utils.py
=============
Caricamento vocabolario ingredienti e parsing dei nomi file/modello.
"""
import os
import csv
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_vocabulary(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    vocab = set()
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ing = row.get('ingr', '').strip().lower()
            if ing:
                vocab.add(ing)
    return list(vocab)


def parse_model_name(name: str):
    """Estrae (gruppo hardware, variante/strategia) dal nome file."""
    name_lower = name.lower()
    group = "Unknown"
    if "e2b" in name_lower and "q4" in name_lower:
        group = "E2B - Q4"
    elif "e2b" in name_lower and "q8" in name_lower:
        group = "E2B - Q8"
    elif "e4b" in name_lower and "q4" in name_lower:
        group = "E4B - Q4"
    elif "e4b" in name_lower and "q8" in name_lower:
        group = "E4B - Q8"

    if "optuna" in name_lower and "yolo" in name_lower:
        variant = "YOLO Optuna"
    elif "v2hybrid" in name_lower:
        variant = "Hybrid v2"
    elif "hybrid" in name_lower:
        variant = "Hybrid v1"
    elif "twostep" in name_lower:
        variant = "Two-Step"
    elif "yolo" in name_lower:
        variant = "YOLO"
    else:
        variant = "Standard"

    return group, variant


def get_dynamic_variants_and_colors(base_dirs=("data/prediction/one_example",
                                                 "data/prediction/one_example_rag")):
    """Scopre dinamicamente le varianti presenti sul disco e assegna un colore."""
    files = []
    for d in base_dirs:
        files = glob.glob(os.path.join(d, "*.json"))
        if files:
            break
    requested = [f for f in files if "optuna" not in f.lower()]
    var_set = set()
    for f in requested:
        filename = os.path.basename(f)
        _, v = parse_model_name(filename.replace('test_predictions_nutrition5k_', '').replace('.json', ''))
        if v != "Unknown" and "Optuna" not in v:
            var_set.add(v)

    variants = sorted(list(var_set))
    cmap = plt.get_cmap('tab10')
    colors = {v: mcolors.to_hex(cmap(i)) for i, v in enumerate(variants)}
    return variants, colors


def build_few_shot_variants(variants):
    """Costruisce l'elenco 'N Example (Strategia)' a partire dalle varianti base."""
    color_map = {
        'Standard': ['#99c2ff', '#3385ff', '#0047b3'],
        'Two-Step': ['#ffb3b3', '#ff4d4d', '#b30000'],
        'Hybrid v1': ['#a3e0a3', '#47d147', '#1f7a1f'],
        'Hybrid v2': ['#d1b3ff', '#9933ff', '#4d0099'],
        'YOLO': ['#ffcc99', '#ff9933', '#cc6600'],
    }
    fs_vars = []
    fs_colors = {}
    for p_idx, p in enumerate(['1 Example', '5 Examples', '10 Examples']):
        for v in variants:
            for key in ('Standard', 'Two-Step', 'Hybrid v2', 'Hybrid v1', 'YOLO'):
                if key in v and (key != 'YOLO' or 'Optuna' not in v):
                    name = f"{p} ({key})"
                    fs_vars.append(name)
                    fs_colors[name] = color_map[key][p_idx]
                    break
    return list(dict.fromkeys(fs_vars)), fs_colors
