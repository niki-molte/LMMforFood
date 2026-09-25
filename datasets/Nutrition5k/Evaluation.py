"""
main.py
=======
Entry point CLI.

Genera:
  - le TABELLE riassuntive a terminale (per-modello e few-shot)
  - lo studio SOGLIA
  - i grafici FEW-SHOT (NLP + Macro), sia per i file "normali" che per i file
    "rag", includendo sempre le varianti overall / exact_match
  - il confronto strategie Optuna (con le relative TABELLE per strategia) e
    il confronto YOLO vs YOLO Optuna

IMPORTANTE: se tutte le cache (CSV/JSON) sono già presenti su disco, nessun
ricalcolo viene eseguito e il modello NLP (SentenceTransformer) non viene
nemmeno caricato: il caricamento è "pigro" (vedi eval/resources.py) e
avviene solo alla prima valutazione effettivamente necessaria.

NOTA SU ROBUSTEZZA: ogni step è avvolto in un try/except. Se uno step fallisce
(es. file mancante, cache corrotta), lo stampiamo per esteso ma continuiamo
con gli step successivi, così un problema nel few-shot RAG, ad esempio, non
impedisce mai al confronto strategie Optuna (e alle sue tabelle) di essere
eseguito.
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
    p = argparse.ArgumentParser(description="Valutazione few-shot (normale/rag) di Nutrition5k.")
    p.add_argument("--few-shot-mode", choices=["normal", "rag", "both"], default="both")
    p.add_argument("--optuna-strategy", default="linear_all5")
    p.add_argument("--optuna-comparison-mode", choices=["none", "rag", "nonrag", "both"], default="both")
    args, _ = p.parse_known_args()
    return args


def run_step(step_name, fn, *fn_args, **fn_kwargs):
    """Esegue uno step della pipeline isolandolo: se fallisce, stampa il
    traceback completo ma NON interrompe l'esecuzione degli step successivi."""
    print(f"\n{'#' * 70}\n# STEP: {step_name}\n{'#' * 70}")
    try:
        return fn(*fn_args, **fn_kwargs)
    except Exception:
        print(f"\n[!!!] LO STEP '{step_name}' E' FALLITO. Traceback completo:")
        traceback.print_exc()
        print(f"[!!!] Continuo con gli step successivi...\n")
        return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"[INFO] Working directory impostata a: {script_dir}")

    args = parse_args()
    print(f"[INFO] Strategia Optuna selezionata: '{args.optuna_strategy}'")
    print(f"[INFO] Modalità confronto strategie Optuna: '{args.optuna_comparison_mode}'")

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

    # Risorse pigre: vocabolario/modello/embeddings vengono caricati solo se
    # e quando serve davvero valutare un file non ancora presente in cache.
    resources = LazyResources(INGREDIENTS_CSV_PATH)

    variants, variant_colors = get_dynamic_variants_and_colors()
    few_shot_variants, _ = build_few_shot_variants(variants)

    # --- 1) Valutazione per-modello: tabelle ---
    eval_csv_path = os.path.join(OUTPUT_DIR, "nlp_evaluation.csv")
    result = run_step(
        "Valutazione per-modello (nlp_evaluation)",
        runners.evaluate_single_run_models,
        resources, THRESHOLD, PRED_JSON_PATHS, eval_csv_path, similarity_dir=SIMILARITY_DIR,
    )
    if result is not None:
        all_nlp_results, all_macro_results = result
        run_step(
            "Stampa tabella riassuntiva per-modello",
            tables.print_summary_tables,
            all_nlp_results, all_macro_results, variants,
        )

    # --- 2) Studio soglia ---
    thresholds = list(np.linspace(1.0, 0.0, 11))
    run_step(
        "Studio soglia (threshold study)",
        runners.evaluate_threshold_study,
        resources, PRED_JSON_PATHS, thresholds, THRESHOLD_DIR,
    )

    # --- 3) Few-shot (1, 5, 10 esempi): NLP + Macro, overall e exact_match ---
    fs_results_normal = None
    fs_results_rag = None

    if run_normal:
        fs_results_normal = run_step(
            "Few-shot (file normali)",
            runners.evaluate_few_shot_learning,
            resources, THRESHOLD, few_shot_variants,
            folders=["one_example", "five_example", "ten_example"],
            folder_to_prefix={"one_example": "1 Example", "five_example": "5 Examples", "ten_example": "10 Examples"},
            output_dir=os.path.join(OUTPUT_DIR, "few_shot_metrics"),
            csv_suffix="", run_label="File normali: 1, 5, 10 Examples",
            optuna_strategy=args.optuna_strategy,
        )

    if run_rag:
        fs_results_rag = run_step(
            "Few-shot (file RAG)",
            runners.evaluate_few_shot_learning,
            resources, THRESHOLD, few_shot_variants,
            folders=["one_example_rag", "five_example_rag", "ten_example_rag"],
            folder_to_prefix={"one_example_rag": "1 Example", "five_example_rag": "5 Examples",
                              "ten_example_rag": "10 Examples"},
            output_dir=os.path.join(OUTPUT_DIR, "few_shot_metrics_rag"),
            csv_suffix="_rag", run_label="File RAG: 1, 5, 10 Examples",
            optuna_strategy=args.optuna_strategy,
        )

    if fs_results_normal:
        run_step(
            "Confronto tecniche (Standard vs YOLO vs YOLO Optuna) - normale",
            runners.evaluate_prompting_technique_comparison,
            fs_results_normal, os.path.join(OUTPUT_DIR, "prompting_technique_comparison"),
        )
    if fs_results_rag:
        run_step(
            "Confronto tecniche (Standard vs YOLO vs YOLO Optuna) - RAG",
            runners.evaluate_prompting_technique_comparison,
            fs_results_rag, os.path.join(OUTPUT_DIR, "prompting_technique_comparison_rag"),
            comparison_label="Standard vs YOLO vs YOLO Optuna (RAG)",
        )

    print(f"\nTutti gli step sono stati eseguiti (vedi sopra per eventuali step falliti). "
          f"Output in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()