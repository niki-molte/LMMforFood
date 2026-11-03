"""
Prompt strategies registry + startup selector for the Nutrition5k extraction scripts.

Every extraction script exposes one or more "axes" (e.g. ``step1``, ``step2``), each one
with several named options. The option to use for each axis is resolved in this order:

    1. explicit argument      -> ``main(step1="weights_only")``      (e.g. from a Kaggle notebook cell)
    2. command line           -> ``python script.py --step1 weights_only``
    3. environment variable   -> ``PROMPT_STEP1=weights_only``
    4. interactive menu       -> only if a terminal / notebook kernel is available
    5. default value          -> used silently when nothing else is available (e.g. Kaggle "Save & Run All")

Option values can be given either by name or by their menu number.
Unknown arguments (e.g. the ``-f kernel.json`` injected by Jupyter/Kaggle) are ignored.
"""

import argparse
import os
import sys

# ==========================================
# --- SINGLE STEP (ExtractFoodInformation.py) ---
# ==========================================
SINGLE_STEP_AXES = {
    "mode": {
        "baseline": "Image only, plain prompt (no YOLO).",
        "yolo": "Image annotated with YOLO boxes + detection report in the prompt.",
    },
}

# ==========================================
# --- TWO STEP (ExtractFoodInformationTwoStep.py) ---
# ==========================================
TWO_STEP_STEP1 = {
    "weights_only": {
        "description": "Visual analysis: ingredients + weights + allergens, NO macronutrients.",
        "schema": "step1_only",
        "system": (
            "You are an expert food vision analyst.\n"
            "Your task is to analyze top-down RGB images of dishes to identify ingredients and portions.\n\n"
            "CORE DIRECTIVES FOR STEP 1:\n"
            "1. Identify all visible ingredients.\n"
            "2. Deduce logically necessary hidden ingredients (e.g., cooking oils, butter, salt, sauces).\n"
            "3. Extract all possible allergens based on visible and hidden ingredients.\n"
            "4. Estimate realistic portion sizes (weight in grams) for each item.\n"
            "5. DO NOT calculate macronutrients. Focus ONLY on identification and weight estimation.\n"
            "You MUST respond ONLY with a valid JSON."
        ),
        "prompt": (
            "Analyze this dish. Provide the full breakdown of visible/hidden ingredients, "
            "estimated weights, and allergens. Output ONLY the JSON."
        ),
    },
}

TWO_STEP_STEP2 = {
    "nutrition": {
        "description": "Compute macronutrients from the Step 1 ingredient list.",
        "system": (
            "You are an expert nutritionist.\n"
            "Your task is to perform a precise nutritional breakdown based on the provided ingredient list and weights.\n\n"
            "CORE DIRECTIVES FOR STEP 2:\n"
            "1. Calculate exact macronutrients (calories, fat, carbohydrates, protein) for each provided ingredient.\n"
            "You MUST respond ONLY with a valid JSON."
        ),
        "prompt": (
            "Analyze this dish with the following extracted details. "
            "Calculate exact macronutrients based on the estimated weights.\n\n"
            "Extracted Food Details:\n"
            "{step1_json}\n\n"
            "Output ONLY the final JSON."
        ),
    },
}

TWO_STEP_AXES = {
    "step1": {k: v["description"] for k, v in TWO_STEP_STEP1.items()},
    "step2": {k: v["description"] for k, v in TWO_STEP_STEP2.items()},
}

# ==========================================
# --- TWO STEP HYBRID (ExtractFoodInformationTwoStepHybrid.py) ---
# "{schema}" is replaced with the JSON schema of NutritionExtraction.
# "{step1_json}" is replaced with the JSON produced by Step 1.
# ==========================================
HYBRID_STEP1 = {
    "full_nutrition": {
        "description": "Step 1 already outputs the full nutritional JSON (macronutrients included).",
        "schema": "full",
        "system": (
            "You are an expert nutritionist and food vision analyst. "
            "Your task is to analyze top-down RGB images of dishes to perform a precise nutritional breakdown.\n\n"
            "CORE DIRECTIVES:\n"
            "1. Identify all visible ingredients.\n"
            "2. Deduce logically necessary hidden ingredients (e.g., cooking oils, butter, salt, sauces) based on the preparation method of the visible food.\n"
            "3. Extract all the possibile allergenes (e.g. gluten, dairy products) based on the visible and hidden ingredients extracted before.\n"
            "4. Estimate realistic portion sizes (weight in grams) for each identified item.\n"
            "5. Calculate exact macronutrients based on the estimated weights.\n"
            "You MUST respond ONLY with a valid JSON strictly adhering to this schema:\n"
            "{schema}"
        ),
        "prompt": (
            "Analyze this dish. Provide the full breakdown of visible/hidden ingredients, "
            "estimated weights, nutritional values and allergens. Output ONLY the JSON."
        ),
    },
    "weights_only": {
        "description": "Step 1 outputs only ingredients + weights + allergens (no macronutrients).",
        "schema": "step1_only",
        "system": (
            "You are an expert food vision analyst.\n"
            "Your task is to analyze top-down RGB images of dishes to identify ingredients and portions.\n\n"
            "CORE DIRECTIVES FOR STEP 1:\n"
            "1. Identify all visible ingredients.\n"
            "2. Deduce logically necessary hidden ingredients (e.g., cooking oils, butter, salt, sauces).\n"
            "3. Extract all possible allergens based on visible and hidden ingredients.\n"
            "4. Estimate realistic portion sizes (weight in grams) for each item.\n"
            "5. DO NOT calculate macronutrients. Focus ONLY on identification and weight estimation.\n"
            "You MUST respond ONLY with a valid JSON."
        ),
        "prompt": (
            "Analyze this dish. Provide the full breakdown of visible/hidden ingredients, "
            "estimated weights, and allergens. Output ONLY the JSON."
        ),
    },
}

HYBRID_STEP2 = {
    "auditor": {
        "description": "Strict auditor: critique and correct the Step 1 extraction against the image.",
        "system": (
            "You are an expert AI culinary judge and strict nutritionist auditor.\n"
            "Your task is to critically evaluate and correct a preliminary nutritional extraction by comparing it against the dish image.\n\n"
            "CORE DIRECTIVES FOR STEP 2 (CRITIQUE & REVISE):\n"
            "1. VISUAL AUDIT: Look closely at the image. Did the preliminary extraction hallucinate ingredients that are not there? Did it miss obvious components?\n"
            "2. VOLUME AUDIT: Are the estimated weights realistic for the plate's proportions?\n"
            "3. MATH AUDIT: Are the macronutrients mathematically sound for those ingredients and weights?\n"
            "4. CORRECT & REFINE: Generate a final, corrected extraction. You can add missing items, remove wrong ones, and adjust all numbers.\n"
            "You MUST respond ONLY with a valid JSON strictly adhering to the exact same schema."
        ),
        "prompt": (
            "Act as a strict auditor. Review the image of this dish and critically evaluate the preliminary extraction below.\n\n"
            "Preliminary Extraction:\n"
            "{step1_json}\n\n"
            "Fix any hallucinations, missed ingredients, unrealistic weights, or incorrect macronutrient calculations. "
            "Output ONLY the final, corrected JSON."
        ),
    },
    "visual_anchor": {
        "description": "Visually anchor the Step 1 portions on the image, then compute macronutrients.",
        "system": (
            "You are an expert nutritionist and food vision analyst.\n"
            "Your task is to perform a precise nutritional breakdown by visually analyzing the image of the dish alongside a provided list of ingredients.\n\n"
            "CORE DIRECTIVES FOR STEP 2:\n"
            "1. LOOK AT THE IMAGE to assess the actual volume, density, and portion size of each item.\n"
            "2. Use the provided ingredient list and preliminary weights as a baseline, but you MUST visually validate and adjust the mass based on the real image.\n"
            "3. Calculate exact macronutrients (calories, fat, carbohydrates, protein) based on your combined visual and textual analysis.\n"
            "You MUST respond ONLY with a valid JSON."
        ),
        "prompt": (
            "Look carefully at the image of this dish and compare it with the following preliminary extraction. "
            "Use the image to visually anchor the actual portion sizes, then calculate the exact macronutrients for the dish.\n\n"
            "Preliminary Food Details:\n"
            "{step1_json}\n\n"
            "Output ONLY the final JSON."
        ),
    },
}

HYBRID_AXES = {
    "step1": {k: v["description"] for k, v in HYBRID_STEP1.items()},
    "step2": {k: v["description"] for k, v in HYBRID_STEP2.items()},
}


# ==========================================
# --- SELECTOR ---
# ==========================================
def _is_interactive() -> bool:
    """True if we can reasonably ask the user something (terminal or Jupyter/Kaggle kernel)."""
    if "ipykernel" in sys.modules:
        return True
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


def _resolve_choice(raw: str, options: list):
    """Accepts an option name or its 1-based menu number. Returns the option or None."""
    raw = raw.strip()
    if raw in options:
        return raw
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1]
    return None


def select_strategy(axes: dict, defaults: dict = None, overrides: dict = None, title: str = "") -> dict:
    """
    Resolves which option to use for every axis (see module docstring for the priority order).

    Args:
        axes: ``{axis_name: {option_name: description}}``.
        defaults: ``{axis_name: option_name}``; falls back to the first option of the axis.
        overrides: explicit choices (e.g. coming from ``main(step1=...)``); ``None`` values are ignored.
        title: heading printed above the menu.

    Returns:
        ``{axis_name: option_name}``.
    """
    defaults = defaults or {}
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}

    parser = argparse.ArgumentParser(add_help=False)
    for axis in axes:
        parser.add_argument(f"--{axis}", default=None)
    cli_args, _ = parser.parse_known_args()  # unknown args (Jupyter/Kaggle's -f ...) are ignored

    chosen = {}
    interactive = _is_interactive()
    printed_title = False

    for axis, options_desc in axes.items():
        options = list(options_desc)
        default = defaults.get(axis, options[0])
        source = None
        value = None

        for src, raw in (
                ("argument", overrides.get(axis)),
                ("command line", getattr(cli_args, axis)),
                ("environment", os.environ.get(f"PROMPT_{axis.upper()}")),
        ):
            if raw is None or str(raw).strip() == "":
                continue
            value = _resolve_choice(str(raw), options)
            if value is None:
                raise ValueError(f"Invalid {src} value '{raw}' for '{axis}'. Valid options: {options}")
            source = src
            break

        if value is None and len(options) > 1 and interactive:
            if title and not printed_title:
                print(f"\n=== {title} ===")
                printed_title = True
            print(f"\nSelect '{axis}':")
            for i, name in enumerate(options, 1):
                tag = " (default)" if name == default else ""
                print(f"  [{i}] {name}{tag} - {options_desc[name]}")
            try:
                while value is None:
                    raw = input(f"{axis} [{default}]: ")
                    if raw.strip() == "":
                        value, source = default, "default"
                        break
                    value = _resolve_choice(raw, options)
                    if value is None:
                        print("  Invalid choice, try again.")
                    else:
                        source = "interactive"
            except (EOFError, KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    raise
                value, source = default, "default (input unavailable)"

        if value is None:
            value, source = default, "default"

        chosen[axis] = value
        print(f"[*] Strategy '{axis}': {value}  <- {source}")

    return chosen
