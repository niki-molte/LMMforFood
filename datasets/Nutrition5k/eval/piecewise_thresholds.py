"""
piecewise_thresholds.py
=======================
Dot-plots (same style as the prompting_comparison charts) of the THRESHOLD
values chosen by Optuna for the `piecewise_all5` strategy.

One panel per nutrient (mass, calories, fat, carbohydrates, protein); X axis =
model groups, 3 points per group (1/5/10 examples), one series per prompting
strategy (Standard / Two-Step / YOLO, lighter color = Semantic Matching).

Usage:  python -m eval.piecewise_thresholds
"""
import os
import json
import re
import matplotlib
matplotlib.use('Agg')

from . import style as st
from .dot_plot import plot_dot_grid

OPTUNA_DIR = "data/optuna"
OUTPUT_DIR = "data/evaluation/few_shot_metrics/piecewise_thresholds"
STRATEGY_DIR = "piecewise_all5"

# (strategy label, optuna folder, folder suffix for the semantic matching runs)
FAMILIES = [
    ("Standard", "few_shot_standard", "semantic_matching"),
    ("Two-Step", "few_shot_twostep", "semantic"),
    ("YOLO", "few_shot_yolo", "semantic_matching"),
]
EXAMPLE_DIRS = {"1 Example": "one_example", "5 Examples": "five_example",
                "10 Examples": "ten_example"}

# plot field -> key in optimal_scalars
FIELD_KEY = {"mass": "mass", "calories": "cal", "fat": "fat",
             "carbohydrates": "carbs", "protein": "protein"}
UNITS = {"mass": "g", "calories": "kcal", "fat": "g", "carbohydrates": "g", "protein": "g"}


def _group_from_filename(name):
    m = re.search(r"(e[24]b)_(q[48])", name.lower())
    if not m:
        return None
    return f"{m.group(1).upper()} - {m.group(2).upper()}"


def load_thresholds():
    """results[group][f'{example} ({series})'][field] = threshold"""
    results = {g: {} for g in st.GROUPS}
    for label, folder, sem_suffix in FAMILIES:
        for matching in (False,):
            series = f"{label} Matching" if matching else label
            for ex_label, ex_dir in EXAMPLE_DIRS.items():
                d = ex_dir + (f"_{sem_suffix}" if matching else "")
                path = os.path.join(OPTUNA_DIR, folder, d, STRATEGY_DIR, "optuna_summary.json")
                if not os.path.exists(path):
                    print(f"[WARNING] missing {path}")
                    continue
                with open(path, encoding="utf-8") as f:
                    summary = json.load(f)
                for fname, entry in summary.items():
                    if fname.startswith("_"):
                        continue
                    g = _group_from_filename(fname)
                    if g not in results:
                        continue
                    sc = entry.get("optimal_scalars", {})
                    results[g][f"{ex_label} ({series})"] = {
                        field: sc.get(f"threshold_{key}") for field, key in FIELD_KEY.items()
                    }
    return results


def plot_thresholds(results, output_dir=OUTPUT_DIR):
    strategies = ['Standard', 'Two-Step', 'YOLO']
    legend_strats = strategies

    def value_fn(group, example, strategy, field):
        res = results.get(group, {}).get(f"{example} ({strategy})")
        return res.get(field) if res else None

    for log_scale in (False, True):
        suffix = "_log" if log_scale else ""
        plot_dot_grid(
            fields=st.MACRO_FIELDS,
            value_fn=value_fn,
            strategies=strategies,
            colors=st.STRATEGY_COLORS,
            markers=st.STRATEGY_MARKERS,
            out_path_combined=os.path.join(output_dir, f"piecewise_thresholds{suffix}_combined.png"),
            out_path_single_template=os.path.join(output_dir, f"piecewise_thresholds{suffix}_{{field}}.png"),
            title_fn=lambda f: f"Optuna Threshold - {st.FIELD_LABELS_IT[f]}",
            ylabel_fn=lambda f: f"Threshold ({UNITS[f]})",
            baseline_fn=None,
            log_scale=log_scale,
            connect_lines=True,
            legend_strategies=legend_strats,
            suptitle="Piecewise All5 - Optuna Thresholds" + (" (Log Scale)" if log_scale else ""),
            legend_ncol=len(legend_strats),
        )


if __name__ == "__main__":
    plot_thresholds(load_thresholds())
    print(f"Saved in {OUTPUT_DIR}")
