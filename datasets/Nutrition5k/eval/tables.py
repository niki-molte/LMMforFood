"""
tables.py
=========
Summary tables printed to the terminal during the evaluation.
"""
from . import style as st
from .data_utils import parse_model_name


def print_summary_tables(all_nlp_results: dict, all_macro_results: dict, variants: list):
    """Per-model table (hardware group x strategy)."""
    print("\n" + "=" * 120)
    print(" " * 45 + "METRICS SUMMARY TABLE")
    print("=" * 120)

    models = list(all_macro_results.keys())
    models_sorted = []
    variant_order = list(variants) + ["Standard Optuna", "YOLO Optuna"]
    for g in st.GROUPS:
        for v in variant_order:
            for m in models:
                mg, mv = parse_model_name(m)
                if mg == g and mv == v:
                    models_sorted.append(m)

    if all_nlp_results:
        print("\n--- NLP METRICS ---")
        header = f"{'Group':<10} | {'Variant':<25} | {'F1':<6} | {'F1 (Std)':<8} | {'IoU':<6} | {'IoU (Std)':<9}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for mod in models_sorted:
            if mod in all_nlp_results:
                g, v = parse_model_name(mod)
                res = all_nlp_results[mod]
                f1_ns = res["Non-Standardized"]["F1"]
                f1_s = res["Standardized"]["F1"]
                iou_ns = res["Non-Standardized"]["IoU"]
                iou_s = res["Standardized"]["IoU"]
                print(f"{g:<10} | {v:<25} | {f1_ns:.4f} | {f1_s:.4f}   | {iou_ns:.4f} | {iou_s:.4f}")
        print("-" * len(header))

    if all_macro_results:
        for subset in ["overall", "exact_match"]:
            title = "MACRONUTRIENTS METRICS (OVERALL)" if subset == "overall" else "MACRONUTRIENTS METRICS (EXACT MATCH)"
            print(f"\n--- {title} ---")
            header = (f"{'Group':<10} | {'Variant':<25} | {'Calories':<15} | {'Mass':<15} | "
                      f"{'Fat':<15} | {'Carbs':<15} | {'Protein':<15}")
            print("-" * len(header))
            print(header)
            print("-" * len(header))

            if subset == "overall":
                b = st.BASELINE
                cal_str = f"{b['calories_MAE']:.1f} ({b['calories_MAE_%']:.1f}%)"
                mass_str = f"{b['mass_MAE']:.1f} ({b['mass_MAE_%']:.1f}%)"
                fat_str = f"{b['fat_MAE']:.1f} ({b['fat_MAE_%']:.1f}%)"
                carb_str = f"{b['carbohydrates_MAE']:.1f} ({b['carbohydrates_MAE_%']:.1f}%)"
                prot_str = f"{b['protein_MAE']:.1f} ({b['protein_MAE_%']:.1f}%)"
                print(f"{'Baseline':<10} | {'Nutrition 5k (2D Direct)':<25} | {cal_str:<15} | "
                      f"{mass_str:<15} | {fat_str:<15} | {carb_str:<15} | {prot_str:<15}")
                print("-" * len(header))

            for mod in models_sorted:
                if mod in all_macro_results:
                    g, v = parse_model_name(mod)
                    r = all_macro_results[mod][subset]
                    cal_str = f"{r['calories_MAE']:.1f} ({r['calories_MAE_%']:.1f}%)"
                    mass_str = f"{r['mass_MAE']:.1f} ({r['mass_MAE_%']:.1f}%)"
                    fat_str = f"{r['fat_MAE']:.1f} ({r['fat_MAE_%']:.1f}%)"
                    carb_str = f"{r['carbohydrates_MAE']:.1f} ({r['carbohydrates_MAE_%']:.1f}%)"
                    prot_str = f"{r['protein_MAE']:.1f} ({r['protein_MAE_%']:.1f}%)"
                    print(f"{g:<10} | {v:<25} | {cal_str:<15} | {mass_str:<15} | {fat_str:<15} | "
                          f"{carb_str:<15} | {prot_str:<15}")
            print("-" * len(header))
            print()


def print_optuna_strategy_tables(results: dict, strategies: list):
    """One table for each Optuna strategy with 4 columns:
    Standard | Standard Optuna | YOLO | YOLO Optuna.

    Args:
        results: dict with structure results[group][variant_name]["macro"]["overall"/"exact_match"].
                 Expected keys:
                   - "{prefix} ({st.STANDARD_BASE_KEY})"  → Standard base
                   - "{prefix} ({strategy}_std)"           → Standard + Optuna
                   - "{prefix} ({st.YOLO_BASE_KEY})"       → YOLO base
                   - "{prefix} ({strategy})"               → YOLO + Optuna
        strategies: list of Optuna strategies (e.g. ["linear_all5", "linear_mass_only", ...]).
    """
    for strategy in strategies:
        label = st.OPTUNA_STRATEGY_LABELS.get(strategy, strategy)
        print(f"\n{'=' * 150}")
        print(f"{' ' * 40}OPTUNA STRATEGY: {label.upper()}")
        print("=" * 150)

        for subset, subset_title in [("overall", "OVERALL MAE & MAE %"),
                                     ("exact_match", "EXACT MATCH MAE & MAE %")]:
            print(f"\n--- {subset_title} ---")
            header = (
                f"{'Group':<12} | {'Variant':<25} | "
                f"{'Calories':<18} | {'Mass':<18} | "
                f"{'Fat':<18} | {'Carbs':<18} | {'Protein':<18}"
            )
            sep = "-" * len(header)
            print(sep)
            print(header)
            print(sep)

            def _fmt(entry):
                if entry is None or "macro" not in entry or subset not in entry["macro"]:
                    return "N/A", "N/A", "N/A", "N/A", "N/A"
                m = entry["macro"][subset]
                return (
                    f"{m.get('calories_MAE', 0.0):.1f} ({m.get('calories_MAE_%', 0.0):.1f}%)",
                    f"{m.get('mass_MAE', 0.0):.1f} ({m.get('mass_MAE_%', 0.0):.1f}%)",
                    f"{m.get('fat_MAE', 0.0):.1f} ({m.get('fat_MAE_%', 0.0):.1f}%)",
                    f"{m.get('carbohydrates_MAE', 0.0):.1f} ({m.get('carbohydrates_MAE_%', 0.0):.1f}%)",
                    f"{m.get('protein_MAE', 0.0):.1f} ({m.get('protein_MAE_%', 0.0):.1f}%)",
                )

            col_specs = [
                ("Standard",         lambda g, p, s=strategy: results.get(g, {}).get(f"{p} ({st.STANDARD_BASE_KEY})")),
                ("Standard Optuna",  lambda g, p, s=strategy: results.get(g, {}).get(f"{p} ({s}_std)")),
                ("YOLO",             lambda g, p, s=strategy: results.get(g, {}).get(f"{p} ({st.YOLO_BASE_KEY})")),
                ("YOLO Optuna",      lambda g, p, s=strategy: results.get(g, {}).get(f"{p} ({s})")),
            ]

            for g in st.GROUPS:
                for prefix in st.EXAMPLES:
                    for col_label, getter in col_specs:
                        variant_label = f"{prefix} ({col_label})"
                        cal, mass, fat, carbs, prot = _fmt(getter(g, prefix))
                        print(
                            f"{g:<12} | {variant_label:<25} | "
                            f"{cal:<18} | {mass:<18} | "
                            f"{fat:<18} | {carbs:<18} | {prot:<18}"
                        )

            print(sep)




def print_few_shot_summary_tables(all_few_shot_results: dict, few_shot_variants: list):
    """Few-shot table (NLP + Macro overall/exact_match)."""
    print("\n" + "=" * 120)
    print(" " * 45 + "FEW-SHOT SUMMARY TABLE")
    print("=" * 120)

    print("\n--- FEW-SHOT NLP METRICS ---")
    header_nlp = (f"{'Group':<12} | {'Variant':<25} | {'F1 (Non-Std)':<12} | {'F1 (Std)':<10} | "
                  f"{'IoU (Non-Std)':<13} | {'IoU (Std)':<10}")
    print("-" * len(header_nlp))
    print(header_nlp)
    print("-" * len(header_nlp))
    for g in st.GROUPS:
        for v in few_shot_variants:
            res = all_few_shot_results.get(g, {}).get(v)
            if res and "nlp" in res and "Standardized" in res["nlp"]:
                f1_s = res["nlp"]["Standardized"]["F1"]
                iou_s = res["nlp"]["Standardized"]["IoU"]
                f1_ns = res["nlp"]["Non-Standardized"]["F1"]
                iou_ns = res["nlp"]["Non-Standardized"]["IoU"]
                print(f"{g:<12} | {v:<25} | {f1_ns:<12.4f} | {f1_s:<10.4f} | {iou_ns:<13.4f} | {iou_s:<10.4f}")
            else:
                print(f"{g:<12} | {v:<25} | {'N/A':<12} | {'N/A':<10} | {'N/A':<13} | {'N/A':<10}")
    print("-" * len(header_nlp))

    header_macro = (f"{'Group':<12} | {'Variant':<25} | {'Calories':<18} | {'Mass':<18} | "
                     f"{'Fat':<18} | {'Carbs':<18} | {'Protein':<18}")

    for subset, title in [("overall", "Overall MAE & MAE %"), ("exact_match", "Exact Match MAE & MAE %")]:
        print(f"\n--- FEW-SHOT MACRO METRICS ({title}) ---")
        print("-" * len(header_macro))
        print(header_macro)
        print("-" * len(header_macro))
        for g in st.GROUPS:
            for v in few_shot_variants:
                res = all_few_shot_results.get(g, {}).get(v)
                if res and "macro" in res and subset in res["macro"]:
                    m = res["macro"][subset]
                    cal = f"{m.get('calories_MAE', 0.0):.1f} ({m.get('calories_MAE_%', 0.0):.1f}%)"
                    mass = f"{m.get('mass_MAE', 0.0):.1f} ({m.get('mass_MAE_%', 0.0):.1f}%)"
                    fat = f"{m.get('fat_MAE', 0.0):.1f} ({m.get('fat_MAE_%', 0.0):.1f}%)"
                    carbs = f"{m.get('carbohydrates_MAE', 0.0):.1f} ({m.get('carbohydrates_MAE_%', 0.0):.1f}%)"
                    prot = f"{m.get('protein_MAE', 0.0):.1f} ({m.get('protein_MAE_%', 0.0):.1f}%)"
                    print(f"{g:<12} | {v:<25} | {cal:<18} | {mass:<18} | {fat:<18} | {carbs:<18} | {prot:<18}")
                else:
                    print(f"{g:<12} | {v:<25} | {'N/A':<18} | {'N/A':<18} | {'N/A':<18} | {'N/A':<18} | {'N/A':<18}")
        print("-" * len(header_macro))


def print_prompting_technique_comparison_table(results: dict):
    """Single table for Standard, Standard Optuna, YOLO and YOLO Optuna.

    The format follows print_few_shot_summary_tables: one row for each
    (Group, Examples, Variant) combination with separate columns for
    Calories, Mass, Fat, Carbs, Protein.
    """
    strategies = ["Standard", "Standard Optuna", "YOLO", "YOLO Optuna"]
    print("\n" + "=" * 150)
    print(" " * 48 + "PROMPTING + OPTUNA SUMMARY TABLE")
    print("=" * 150)

    for subset, title in [("overall", "OVERALL MAE & MAE %"),
                          ("exact_match", "EXACT MATCH MAE & MAE %")]:
        print(f"\n--- {title} ---")
        header = (
            f"{'Group':<12} | {'Variant':<25} | "
            f"{'Calories':<18} | {'Mass':<18} | "
            f"{'Fat':<18} | {'Carbs':<18} | {'Protein':<18}"
        )
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)

        for g in st.GROUPS:
            for prefix in st.EXAMPLES:
                for strategy in strategies:
                    variant_label = f"{prefix} ({strategy})"
                    entry = results.get(g, {}).get(variant_label)
                    if entry and "macro" in entry and subset in entry["macro"]:
                        m = entry["macro"][subset]
                        cal   = f"{m.get('calories_MAE', 0.0):.1f} ({m.get('calories_MAE_%', 0.0):.1f}%)"
                        mass  = f"{m.get('mass_MAE', 0.0):.1f} ({m.get('mass_MAE_%', 0.0):.1f}%)"
                        fat   = f"{m.get('fat_MAE', 0.0):.1f} ({m.get('fat_MAE_%', 0.0):.1f}%)"
                        carbs = f"{m.get('carbohydrates_MAE', 0.0):.1f} ({m.get('carbohydrates_MAE_%', 0.0):.1f}%)"
                        prot  = f"{m.get('protein_MAE', 0.0):.1f} ({m.get('protein_MAE_%', 0.0):.1f}%)"
                    else:
                        cal = mass = fat = carbs = prot = "N/A"
                    print(
                        f"{g:<12} | {variant_label:<25} | "
                        f"{cal:<18} | {mass:<18} | "
                        f"{fat:<18} | {carbs:<18} | {prot:<18}"
                    )

        print(sep)
