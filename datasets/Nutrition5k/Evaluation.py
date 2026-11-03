"""
main.py
=======
CLI entry point.

Generates:
  - summary TABLES on the terminal (per-model and few-shot)
  - the THRESHOLD study
  - the FEW-SHOT plots (NLP + Macro), for both the "normal" and the "semantic_matching"
    files, always including the overall / exact_match variants
  - the Optuna strategy comparison (with the per-strategy TABLES) and
    the YOLO vs YOLO Optuna comparison

IMPORTANT: if all the caches (CSV/JSON) are already present on disk, no
recomputation is performed and the NLP model (SentenceTransformer) is not
even loaded: loading is lazy (see eval/resources.py) and only happens at the
first evaluation that is actually needed.

NOTE ON ROBUSTNESS: each step is wrapped in a try/except. If a step fails
(e.g. missing file, corrupted cache), the full error is printed but the
following steps still run, so that a problem in the Semantic Matching few-shot step, for
example, never prevents the Optuna strategy comparison (and its tables) from
being executed.
"""
import os
import argparse
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')

from eval import (
    LazyResources, get_dynamic_variants_and_colors, build_few_shot_variants,
    runners, plots, tables,
)


def parse_args():
    p = argparse.ArgumentParser(description="Few-shot evaluation (normal/semantic_matching) of Nutrition5k.")
    p.add_argument("--few-shot-mode", choices=["normal", "semantic_matching", "both"], default="both")
    p.add_argument("--optuna-strategy", default="piecewise_all5")
    p.add_argument("--optuna-comparison-mode", choices=["none", "semantic_matching", "non_semantic_matching", "both"], default="both")
    args, _ = p.parse_known_args()
    return args


def run_step(step_name, fn, *fn_args, **fn_kwargs):
    """Run a pipeline step in isolation: if it fails, print the full
    traceback but do NOT interrupt the execution of the following steps."""
    print(f"\n{'#' * 70}\n# STEP: {step_name}\n{'#' * 70}")
    try:
        return fn(*fn_args, **fn_kwargs)
    except Exception:
        print(f"\n[!!!] STEP '{step_name}' FAILED. Full traceback:")
        traceback.print_exc()
        print(f"[!!!] Continuing with the next steps...\n")
        return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"[INFO] Working directory set to: {script_dir}")

    args = parse_args()
    print(f"[INFO] Selected Optuna strategy: '{args.optuna_strategy}'")
    print(f"[INFO] Optuna strategy comparison mode: '{args.optuna_comparison_mode}'")

    run_normal = args.few_shot_mode in ("normal", "both")
    run_semantic_matching = args.few_shot_mode in ("semantic_matching", "both")

    THRESHOLD = 0.6
    OUTPUT_DIR = "data/evaluation"
    THRESHOLD_DIR = os.path.join(OUTPUT_DIR, "threshold_study")
    SIMILARITY_DIR = os.path.join(OUTPUT_DIR, "similarity_analysis")
    INGREDIENTS_CSV_PATH = "data/metadata/ingredients_metadata.csv"

    ONE_EXAMPLE_DIR = "data/prediction/one_example"
    PRED_JSON_PATHS = [
        os.path.join(ONE_EXAMPLE_DIR, f) for f in sorted(os.listdir(ONE_EXAMPLE_DIR))
        if f.endswith(".json") and "optuna" not in f.lower()
    ] if os.path.isdir(ONE_EXAMPLE_DIR) else []

    # Lazy resources: vocabulary/model/embeddings are loaded only if and when
    # a file not yet in cache actually needs to be evaluated.
    resources = LazyResources(INGREDIENTS_CSV_PATH)

    variants, variant_colors = get_dynamic_variants_and_colors()
    few_shot_variants, _ = build_few_shot_variants(variants)

    # --- 1) Per-model evaluation: tables ---
    eval_csv_path = os.path.join(OUTPUT_DIR, f"nlp_evaluation_{args.optuna_strategy}.csv")
    result = run_step(
        "Per-model evaluation (nlp_evaluation)",
        runners.evaluate_single_run_models,
        resources, THRESHOLD, PRED_JSON_PATHS, eval_csv_path, similarity_dir=SIMILARITY_DIR,
        optuna_strategy=args.optuna_strategy,
    )
    if result is not None:
        all_nlp_results, all_macro_results = result
        run_step(
            "Print per-model summary table",
            tables.print_summary_tables,
            all_nlp_results, all_macro_results, variants,
        )

    # --- 2) Threshold study ---
    thresholds = list(np.linspace(1.0, 0.0, 11))
    run_step(
        "Threshold study",
        runners.evaluate_threshold_study,
        resources, PRED_JSON_PATHS, thresholds, THRESHOLD_DIR,
    )

    # --- 3) Few-shot (1, 5, 10 examples): NLP + Macro, overall and exact_match ---
    fs_results_normal = None
    fs_results_semantic_matching = None

    if run_normal:
        fs_results_normal = run_step(
            "Few-shot (normal files)",
            runners.evaluate_few_shot_learning,
            resources, THRESHOLD, few_shot_variants,
            folders=["one_example", "five_example", "ten_example"],
            folder_to_prefix={"one_example": "1 Example", "five_example": "5 Examples", "ten_example": "10 Examples"},
            output_dir=os.path.join(OUTPUT_DIR, "few_shot_metrics"),
            csv_suffix="", run_label="Normal files: 1, 5, 10 Examples",
            optuna_strategy=args.optuna_strategy,
        )

    if run_semantic_matching:
        fs_results_semantic_matching = run_step(
            "Few-shot (Semantic Matching files)",
            runners.evaluate_few_shot_learning,
            resources, THRESHOLD, few_shot_variants,
            folders=["one_example_semantic_matching", "five_example_semantic_matching", "ten_example_semantic_matching"],
            folder_to_prefix={"one_example_semantic_matching": "1 Example", "five_example_semantic_matching": "5 Examples",
                              "ten_example_semantic_matching": "10 Examples"},
            output_dir=os.path.join(OUTPUT_DIR, "few_shot_metrics_semantic_matching"),
            csv_suffix="_semantic_matching", run_label="Semantic Matching files: 1, 5, 10 Examples",
            optuna_strategy=args.optuna_strategy,
        )

        # --- 4) Optuna strategy comparison: Standard / Standard Optuna / YOLO / YOLO Optuna
        #        One table per strategy, on both normal and Semantic Matching files.
        run_step(
            "Optuna strategy comparison (normal files)",
            runners.evaluate_optuna_strategy_comparison,
            resources, THRESHOLD, semantic_matching=False,
        )
        run_step(
            "Optuna strategy comparison (Semantic Matching files)",
            runners.evaluate_optuna_strategy_comparison,
            resources, THRESHOLD, semantic_matching=True,
        )

        # --- 5) Overall combined plot ---
        run_step(
            "Combined Strategies Plot Generation (All-in-One)",
            runners.evaluate_all_strategies_combined,
            resources, THRESHOLD,
            output_dir=os.path.join(OUTPUT_DIR, "combined_plots"),
            optuna_strategy=args.optuna_strategy
        )

        print(f"\nAll steps have been executed (see above for any failed steps). "
              f"Output in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()