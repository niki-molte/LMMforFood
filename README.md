# LLMforFood

Master's thesis project: **estimating ingredients, weights and nutritional values from food images** with
Vision-Language Models (VLMs), evaluated on the
[Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) dataset.

Models run locally with **Ollama** (Gemma 4, Qwen 3.5, MiniCPM-V 4.5). Responses are constrained to
**Pydantic** JSON schemas and compared against the ground truth (MAE on macronutrients, matching/semantic
similarity on ingredients).

## Pipeline

```text
Nutrition5k ──► Nutrition5k.py (train/test JSON split)
                    │
                    ├─► [opt.] run_yolo_annotate.py (YOLO grounding)
                    ▼
        ExtractFoodInformation*.py (VLM, few-shot) ──► data/prediction/*.json
                    │                                        │
                    └─► optimization/run_optuna.py           ▼
                        (weight calibration)           Evaluation.py ──► tables, plots, data/evaluation/
```

## Structure

```text
LLMforFood/
├── settings.py            # centralized .env loading (get_env / get_path)
├── .env.example           # environment variables template
├── requirements.txt
├── answer_schema/         # Pydantic response schemas
├── OllamaClient/          # base client + Gemma, Qwen, MiniCPM
├── yolo_weights/          # YOLOv8 weights (yolov8n/yolov8s)
├── docs/                  # Sphinx documentation (GitHub Pages)
└── datasets/
    ├── BaseDataset.py     # base split loader
    └── Nutrition5k/
        ├── Nutrition5k.py                          # builds the JSON splits (ground truth)
        ├── AnalyzeNutrition5k.py                   # exploratory dataset analysis
        ├── ExtractFoodInformation.py               # single-step extraction
        ├── ExtractFoodInformationTwoStep.py        # two-step extraction
        ├── ExtractFoodInformationTwoStepHybrid.py  # hybrid two-step
        ├── prompt_strategies.py                    # prompt strategy catalog
        ├── run_yolo_annotate.py                    # YOLO annotation of selected dishes
        ├── Evaluation.py + eval/                   # evaluation, tables and plots
        ├── optimization/                           # Optuna (run_optuna, sample_optuna_subset)
        ├── perfect_prediction.py                   # "perfect" predictions (upper bound)
        ├── finetuning_dataset_extraction.py        # JSONL dataset for Gemma fine-tuning
        ├── inspect_dish.py, count_entries.py       # inspection utilities
        └── data/                                   # ground truth, predictions, evaluations, optuna
```

## 1. Installation

Requirements: Python 3.12, [Ollama](https://ollama.com) running (`localhost:11434`), CUDA GPU recommended
(YOLO, sentence-transformers), Nutrition5k dataset downloaded.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# local models (names must match those in .env)
ollama pull gemma4:e2b
ollama pull qwen3.5:4b
ollama pull openbmb/minicpm-v4.5:8b
```

## 2. Configuration

```bash
cp .env.example .env
```

Set the dataset path and the Ollama model names in `.env`:

```text
NUTRITION5K_DATASET_DIR=/path/to/nutrition5k_dataset   # contains imagery/, metadata/, dish_ids/
OLLAMA_GEMMA_4_E2B_Q4=gemma4:e2b
OLLAMA_QWEN_3.5_4B=qwen3.5:4b
OLLAMA_MINICPM_V4.5_8B=openbmb/minicpm-v4.5:8b
```

The `.env` file is not versioned. In code, variables are read with `settings.get_env()` / `get_path()`.

> [!IMPORTANT]
> Scripts use absolute imports from the project root (`settings`, `OllamaClient`, ...): run them with the root
> in `PYTHONPATH`. In PyCharm, mark `LLMforFood/` as *Sources Root*; from a terminal:
> ```bash
> export PYTHONPATH=$(pwd)
> cd datasets/Nutrition5k
> ```
> Scripts write to `data/...` relative to their own folder, so run them from `datasets/Nutrition5k/`.

## 3. Usage

All the following commands start from `datasets/Nutrition5k/` with `PYTHONPATH` set.

### 3.1 Build the dataset splits

```bash
python Nutrition5k.py
```
Reads the official Nutrition5k metadata and splits and prepares the ground truth in `data/ground truth/`
(`train_dataset.json`, `test_dataset.json`). *Note:* the `save_json()` call is commented out in the script:
uncomment it if you want to (re)write the files.

### 3.2 (Optional) Dataset analysis

```bash
python AnalyzeNutrition5k.py              # also shows the plots
python AnalyzeNutrition5k.py --save-only  # only saves them
```

### 3.3 Extraction with the VLM

Three scripts, configured the same way: constants at the top of the file.

| Script | Description |
|---|---|
| `ExtractFoodInformation.py` | Single-step: one call outputs ingredients, weights and macronutrients |
| `ExtractFoodInformationTwoStep.py` | Step 1: ingredients and weights; Step 2: macronutrients |
| `ExtractFoodInformationTwoStepHybrid.py` | Hybrid two-step with few-shot examples |

Main constants to edit in the script:

| Constant | Meaning |
|---|---|
| `PROVIDER` | `"gemma"`, `"qwen"` or `"minicpm"` |
| `USE_YOLO` | use YOLO grounding (annotated image + report in the prompt) |
| `NUM_EXAMPLES` | number of few-shot examples (1, 5, 10, ...) |
| `MAX_ITEMS_TO_TEST` | number of dishes to process (raise it for the whole test set) |
| `SAVE_EVERY_N_ITEMS` | intermediate save frequency |
| `OUTPUT_PREDICTIONS_PATH` | output JSON file (in `data/prediction/...`) |

```bash
python ExtractFoodInformation.py
```

**Prompt strategies.** Defined in `prompt_strategies.py`. The choice is resolved in this order:
explicit argument → command line → environment variable → interactive menu → default.

```bash
python ExtractFoodInformation.py --mode yolo          # or: baseline
PROMPT_STEP1=weights_only python ExtractFoodInformationTwoStep.py
```
The available axes (`mode`, `step1`, `step2`) and their options are listed in `prompt_strategies.py`
(e.g. `weights_only`, `full_nutrition`, `auditor`, `visual_anchor`).

### 3.4 YOLO grounding (optional)

```bash
python run_yolo_annotate.py                              # uses yolo_weights/yolov8s.pt
python run_yolo_annotate.py --weights ../../yolo_weights/yolov8n.pt --out data/yolo_annotated
```
Edit `DISH_IDS` in the script to choose the dishes to annotate.

### 3.5 Calibration with Optuna

```bash
python optimization/sample_optuna_subset.py            # samples the train subset
OPTUNA_STRATEGY=all python optimization/run_optuna.py  # or a single strategy
```
Strategies (`STRATEGY_CATALOG` in `run_optuna.py`): `linear_all5`, `linear_mass_only`, `linear_bias_all5`,
`power_law_all5`, `piecewise_all5`. Input predictions live in `data/optuna/...`; results
(`optuna_summary.json`) are saved next to them.

### 3.6 Evaluation

```bash
python Evaluation.py
python Evaluation.py --few-shot-mode normal --optuna-strategy piecewise_all5 --optuna-comparison-mode both
```

| Option | Values (default in bold) |
|---|---|
| `--few-shot-mode` | `normal`, `semantic_matching`, **`both`** |
| `--optuna-strategy` | **`piecewise_all5`** (or another strategy) |
| `--optuna-comparison-mode` | `none`, `semantic_matching`, `non_semantic_matching`, **`both`** |

Produces terminal tables, the NLP threshold study, few-shot plots and Optuna comparisons in `data/evaluation/`.
If the CSV/JSON caches already exist, nothing is recomputed (and the NLP model is not even loaded).

### 3.7 Utilities

```bash
python inspect_dish.py dish_1561662216 [--threshold 0.6] [--no-std]  # compare predictions and GT for one dish
python count_entries.py                                              # sanity check on predicted ingredients
python perfect_prediction.py                                         # generate perfect predictions (upper bound)
```
`finetuning_dataset_extraction.py` exposes `jsonl_finetuning_gemma()` to build the JSONL fine-tuning datasets
(`data/ground truth/finetuning_*_data.jsonl`).

## 4. API documentation

The documentation is generated with Sphinx from the docstrings (Furo theme):

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```
The `.github/workflows/docs.yml` workflow publishes it to GitHub Pages (*Settings → Pages → Source: GitHub Actions*).
