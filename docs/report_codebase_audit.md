# Codebase Audit Report: Language, Prints & LLM-Style Comments
> **Scope**: Entire repository (`/home/niki/PycharmProjects/LLMforFood`), strictly **excluding `VireoFood172`** and `.venv`.
> **Target Consumer**: This document is structured for rapid human review and can be passed directly as prompt instructions to an LLM / coding agent for automated remediation.

## 1. Executive Summary
- **Total Python files scanned**: 43 (all files in `LLMforFood/`, excluding `datasets/VireoFood172/` and `.venv/`)
- **Files with Italian Docstrings**: 21 / 43 (~48.8%)
- **Files with Italian Inline/Block Comments**: 20 / 43 (~46.5%)
- **Files with Italian Print Statements**: 18 / 43 (~41.9%)
- **Files with LLM / Temporary Dev Residue Comments**: 5 / 43

### Key Answers to User Questions
1. **Is the code documented in English?**  
   **NO.** Most modules, classes, and function docstrings are written in **Italian** (e.g. `datasets/Nutrition5k/`, `eval/`, `OllamaClient/`, `settings.py`). Only low-level boilerplate or schema definitions are in English.
2. **Are prints all in English?**  
   **NO.** There are ~100+ console output statements across 18 files printed in **Italian** (`[INFO] Working directory impostata a:`, `Caricamento del file in corso...`, `Errore:`, etc.).
3. **Are there LLM-style comments (e.g. `#ho cambiato...`)?**  
   **YES.** Several comments showing LLM/developer temporary modifications exist, notably:
   - `datasets/Nutrition5k/finetuning_dataset_extraction.py:62`: `# Rimosso il filtro: if ing["weight_grams"] < 1.0: continue`
   - `datasets/Nutrition5k/optimization/run_optuna.py:216`: `# NUOVO: calcola il range di soglia specifico per questo nutriente`
   - `datasets/Nutrition5k/ExtractFoodInformationTwoStepHybrid.py:225`: `# Salta il passaggio al LLM per il momento`
   - `datasets/Nutrition5k/ExtractFoodInformationTwoStep.py:228`: `# Salta il passaggio al LLM per il momento` (note: also contains unresolved git conflict markers)
   - `OllamaClient/*_client.py`: `# --- DEEP DEBUG ---` block markers

## 2. File-by-File Summary Table

| File | IT Docstrings | IT Comments | IT Prints | LLM Comments | Action Required |
|---|:---:|:---:|:---:|:---:|---|
| `GoogleLLMClient/__init__.py` | 0 | 0 | 0 | 0 | OK (English / Clean) |
| `GoogleLLMClient/client.py` | 1 | 0 | 0 | 0 | Translate docstrings |
| `GoogleLLMClient/exceptions.py` | 0 | 0 | 0 | 0 | OK (English / Clean) |
| `HuggingFaceLLMClient/__init__.py` | 0 | 0 | 0 | 0 | OK (English / Clean) |
| `HuggingFaceLLMClient/client.py` | 1 | 0 | 0 | 0 | Translate docstrings |
| `OllamaClient/__init__.py` | 0 | 0 | 0 | 0 | OK (English / Clean) |
| `OllamaClient/gemma_client.py` | 4 | 0 | 2 | 0 | Translate docstrings, Translate prints |
| `OllamaClient/miniCPM_client.py` | 5 | 5 | 2 | 0 | Translate docstrings, Translate comments, Translate prints |
| `OllamaClient/ollama_client.py` | 5 | 0 | 0 | 0 | Translate docstrings |
| `OllamaClient/qwen_client.py` | 4 | 2 | 2 | 0 | Translate docstrings, Translate comments, Translate prints |
| `answer_schema/__init__.py` | 0 | 0 | 0 | 0 | OK (English / Clean) |
| `answer_schema/extraction.py` | 0 | 0 | 0 | 0 | OK (English / Clean) |
| `datasets/BaseDataset.py` | 0 | 0 | 0 | 0 | OK (English / Clean) |
| `datasets/Nutrition5k/AnalyzeNutrition5k.py` | 9 | 15 | 10 | 0 | Translate docstrings, Translate comments, Translate prints |
| `datasets/Nutrition5k/Evaluation.py` | 2 | 5 | 3 | 0 | Translate docstrings, Translate comments, Translate prints |
| `datasets/Nutrition5k/ExtractFoodInformation.py` | 3 | 13 | 13 | 0 | Translate docstrings, Translate comments, Translate prints |
| `datasets/Nutrition5k/ExtractFoodInformationTwoStep.py` | 0 | 11 | 10 | 1 | Translate comments, Translate prints, Clean LLM comments, Resolve merge conflicts |
| `datasets/Nutrition5k/ExtractFoodInformationTwoStepHybrid.py` | 0 | 10 | 10 | 1 | Translate comments, Translate prints, Clean LLM comments |
| `datasets/Nutrition5k/Nutrition5k.py` | 0 | 0 | 1 | 0 | Translate prints |
| `datasets/Nutrition5k/__init__.py` | 0 | 0 | 0 | 0 | OK (English / Clean) |
| `datasets/Nutrition5k/count_entries.py` | 0 | 0 | 3 | 0 | Translate prints |
| `datasets/Nutrition5k/data/evaluation/similarity_analysis/find_best_examples.py` | 0 | 0 | 0 | 0 | OK (English / Clean) |
| `datasets/Nutrition5k/eval/__init__.py` | 0 | 0 | 0 | 0 | OK (English / Clean) |
| `datasets/Nutrition5k/eval/data_utils.py` | 2 | 1 | 0 | 0 | Translate docstrings, Translate comments |
| `datasets/Nutrition5k/eval/dot_plot.py` | 2 | 2 | 0 | 0 | Translate docstrings, Translate comments |
| `datasets/Nutrition5k/eval/evaluator.py` | 1 | 0 | 0 | 0 | Translate docstrings |
| `datasets/Nutrition5k/eval/plots.py` | 3 | 12 | 0 | 0 | Translate docstrings, Translate comments |
| `datasets/Nutrition5k/eval/resources.py` | 1 | 0 | 0 | 0 | Translate docstrings |
| `datasets/Nutrition5k/eval/runners.py` | 4 | 14 | 12 | 0 | Translate docstrings, Translate comments, Translate prints |
| `datasets/Nutrition5k/eval/style.py` | 3 | 10 | 0 | 0 | Translate docstrings, Translate comments |
| `datasets/Nutrition5k/eval/tables.py` | 5 | 0 | 2 | 0 | Translate docstrings, Translate prints |
| `datasets/Nutrition5k/finetuning_dataset_extraction.py` | 0 | 2 | 2 | 1 | Translate comments, Translate prints, Clean LLM comments |
| `datasets/Nutrition5k/inspect_dish.py` | 9 | 14 | 8 | 1 | Translate docstrings, Translate comments, Translate prints, Clean LLM comments |
| `datasets/Nutrition5k/optimization/__init__.py` | 0 | 0 | 0 | 0 | OK (English / Clean) |
| `datasets/Nutrition5k/optimization/run_optuna.py` | 7 | 26 | 9 | 1 | Translate docstrings, Translate comments, Translate prints, Clean LLM comments |
| `datasets/Nutrition5k/optimization/sample_optuna_subset.py` | 0 | 9 | 4 | 0 | Translate comments, Translate prints |
| `datasets/Nutrition5k/perfect_prediction.py` | 0 | 0 | 1 | 0 | Translate prints |
| `datasets/Nutrition5k/run_yolo_annotate.py` | 2 | 3 | 4 | 0 | Translate docstrings, Translate comments, Translate prints |
| `datasets/__init__.py` | 0 | 0 | 0 | 0 | OK (English / Clean) |
| `patch_draw.py` | 0 | 5 | 0 | 0 | Translate comments |
| `patch_eval.py` | 0 | 0 | 0 | 0 | OK (English / Clean) |
| `revert_draw.py` | 0 | 1 | 0 | 0 | Translate comments |
| `settings.py` | 4 | 5 | 0 | 0 | Translate docstrings, Translate comments |

## 3. Detailed File Breakdown & Automated Refactoring Instructions

### `GoogleLLMClient/client.py`
**Italian Docstrings to Translate to English:**
- Line 12 (`GoogleClient`): `"A client wrapper for interacting with Google's Generative AI models.
Supports Few-Shot Prompting thr..."`

---

### `HuggingFaceLLMClient/client.py`
**Italian Docstrings to Translate to English:**
- Line 13 (`HuggingFaceClient`): `"A client wrapper for interacting with Hugging Face Inference API via the native huggingface_hub libr..."`

---

### `OllamaClient/gemma_client.py`
**Italian Docstrings to Translate to English:**
- Line 6 (`GemmaClient`): `"Client concreto per interagire con i modelli della famiglia Ollama/Gemma4.
Eredita da BaseVisionClie..."`
- Line 12 (`_define_model_configuration`): `"Costruisce i kwargs di configurazione specifici per Ollama...."`
- Line 31 (`_prepare_contents`): `"Costruisce l'array di messaggi specifico per il formato chat di Ollama...."`
- Line 76 (`_execute`): `"Esegue la chiamata API locale a Ollama...."`

**Italian Print Statements to Translate (2 found):**
- Line 95: `print("[OllamaClient ERRORE] Formato di risposta non riconosciuto.")`
- Line 99: `print("[OllamaClient AVVISO] Il modello ha restituito una risposta vuota.")`

---

### `OllamaClient/miniCPM_client.py`
**Italian Docstrings to Translate to English:**
- Line 6 (`MiniCPMClient`): `"Client concreto per interagire con la famiglia MiniCPM-V (es. minicpm-v4.5).
Eredita da BaseVisionCl..."`
- Line 12 (`__init__`): `"Inizializza il client per MiniCPM.

:param model_name: Nome del modello su Ollama (es. 'minicpm-v4.5..."`
- Line 26 (`_define_model_configuration`): `"Costruisce i kwargs di configurazione. Varia in base al thinking_mode...."`
- Line 54 (`_prepare_contents`): `"Costruisce l'array di messaggi con gli esempi e inietta la direttiva di pensiero...."`
- Line 107 (`_execute`): `"Esegue la chiamata API locale a Ollama con log dinamici...."`

**Italian Comments to Translate (5 found, sampling lines):**
- Line 35: `# Parametri dinamici in base al livello di pensiero richiesto`
- Line 40: `# Spazio di generazione molto ampio per i ragionamenti`
- Line 41: `# Contesto bloccato a 32k come da te richiesto`
- Line 47: `# Taglio rapido per efficienza`
- Line 84: `# L'input di esempio con immagine`

**Italian Print Statements to Translate (2 found):**
- Line 125: `print(f"[{self.__class__.__name__} ERRORE] Formato di risposta non riconosciuto.")`
- Line 129: `print(f"[{self.__class__.__name__} AVVISO] Il modello ha restituito una risposta vuota.")`

---

### `OllamaClient/ollama_client.py`
**Italian Docstrings to Translate to English:**
- Line 6 (`BaseVisionClient`): `"Classe astratta di base (Interfaccia) per tutti i client di visione locale o cloud.
Garantisce che o..."`
- Line 17 (`_define_model_configuration`): `"Costruisce i parametri/kwargs di configurazione specifici per il provider...."`
- Line 22 (`_prepare_contents`): `"Formatta i messaggi, le immagini e gli esempi according al formato richiesto dal provider...."`
- Line 28 (`_execute`): `"Esegue l'effettiva chiamata di inferenza all'API del modello...."`
- Line 32 (`ask`): `"Orchestratore pubblico (Template Method).
Questo flusso rimane identico per qualsiasi modello di vis..."`

---

### `OllamaClient/qwen_client.py`
**Italian Docstrings to Translate to English:**
- Line 5 (`QwenClient`): `"Client concreto per interagire con i modelli della famiglia Qwen (es. qwen3.5:4b, qwen3.5:9b).
Eredi..."`
- Line 12 (`_define_model_configuration`): `"Costruisce i kwargs di configurazione specifici per Qwen su Ollama...."`
- Line 33 (`_prepare_contents`): `"Costruisce l'array di messaggi specifico per il formato chat di Ollama/Qwen...."`
- Line 72 (`_execute`): `"Esegue la chiamata API locale a Ollama...."`

**Italian Comments to Translate (2 found, sampling lines):**
- Line 21: `# Parametri estratti ufficialmente dalla documentazione/Modelfile di Ollama per Qwen 3.5`
- Line 28: `# Finestra di contesto bloccata a 32k`

**Italian Print Statements to Translate (2 found):**
- Line 91: `print(f"[{self.__class__.__name__} ERRORE] Formato di risposta non riconosciuto.")`
- Line 95: `print(f"[{self.__class__.__name__} AVVISO] Il modello ha restituito una risposta vuota.")`

---

### `datasets/Nutrition5k/AnalyzeNutrition5k.py`
**Italian Docstrings to Translate to English:**
- Line 16 (`AnalyzeNutrition5k`): `"Classe per l'analisi esplorativa (EDA) del dataset Nutrition5k, 
unendo i set di training e testing ..."`
- Line 39 (`plot_dish_complexity`): `"1. Complessità dei piatti (Numero di ingredienti)
Istogramma a barre che mostra quanti ingredienti c..."`
- Line 76 (`plot_nutritional_distribution`): `"2. Distribuzione dei Valori Nutrizionali
Grafico che mostra la distribuzione di Calorie (Kcal) e Mas..."`
- Line 103 (`plot_top_ingredients`): `"3. Gli Ingredienti più frequenti
Grafico a barre orizzontali con gli ingredienti più presenti nel da..."`
- Line 130 (`analyze_invisible_ingredients`): `"4. Presenza di ingredienti "invisibili"
Stampa a console alcune statistiche su ingredienti nascosti ..."`
- Line 182 (`save_example_images`): `"5. Salvataggio di immagini singole di esempio + metadati JSON.

Per ogni categoria viene creata una ..."`
- Line 212 (`_build_metadata`): `"Costruisce il dizionario di metadati per un singolo piatto...."`
- Line 246 (`_sample_by_quartiles`): `"Restituisce al massimo n_per_bin righe per ognuno dei 4 quartili...."`
- Line 262 (`_save_category`): `"Salva le immagini singole e il JSON per una categoria...."`

**Italian Comments to Translate (15 found, sampling lines):**
- Line 30: `# Estrazione dei macronutrienti per un accesso più semplice`
- Line 45: `# Calcolo del numero di ingredienti validi per ogni piatto`
- Line 51: `# Suddivisione in categorie: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12+`
- Line 143: `# Esempi di keyword per ingredienti invisibili espansi (come erano nel file prima)`
- Line 159: `# Evitiamo di graficare quelli a zero per chiarezza`
- Line 165: `# Generazione del grafico (prendiamo solo i top 20 per non affollare troppo il grafico)`
- Line 285: `# --- a) Numero di ingredienti ---`
- Line 305: `# --- d) Piatti estremi: >= 15 ingredienti O >= 1000 kcal ---`
- Line 310: `# 2 con molti ingredienti + 2 con molte calorie (evitando duplicati)`
- Line 322: `# 2. Valori nutrizionali`
- *(and 5 more Italian comments in this file)*

**Italian Print Statements to Translate (10 found):**
- Line 156: `print("\nFrequenza di alcuni ingredienti 'invisibili' (o che li contengono) nel dataset:")`
- Line 265: `print(f"  [WARN] Nessun piatto trovato per: {category_name}")`
- Line 286: `print("  Campionamento per numero di ingredienti...")`
- Line 294: `print("  Campionamento per calorie...")`
- Line 300: `print("  Campionamento per massa totale...")`
- Line 330: `print("\nSalvataggio immagini di esempio...")`
- Line 354: `print(f"Caricamento file in corso...\n- Train: {TRAIN_PATH}\n- Test: {TEST_PATH}")`
- Line 355: `print(f"I grafici verranno salvati in: {OUTPUT_DIR}")`
- Line 371: `print(f"\nAnalisi completata con successo! Grafici salvati in {OUTPUT_DIR}")`
- Line 374: `print(f"Errore: {e}")`

---

### `datasets/Nutrition5k/Evaluation.py`
**Italian Docstrings to Translate to English:**
- Line 1 (`Module`): `"main.py
=======
Entry point CLI.

Genera:
  - le TABELLE riassuntive a terminale (per-modello e few-..."`
- Line 47 (`run_step`): `"Esegue uno step della pipeline isolandolo: se fallisce, stampa il
traceback completo ma NON interrom..."`

**Italian Comments to Translate (5 found, sampling lines):**
- Line 84: `# Risorse pigre: vocabolario/modello/embeddings vengono caricati solo se`
- Line 85: `# e quando serve davvero valutare un file non ancora presente in cache.`
- Line 91: `# --- 1) Valutazione per-modello: tabelle ---`
- Line 107: `# --- 2) Studio soglia ---`
- Line 145: `#        Una tabella per ogni strategia, sia su file normali che RAG.`

**Italian Print Statements to Translate (3 found):**
- Line 54: `print(f"\n[!!!] LO STEP '{step_name}' E' FALLITO. Traceback completo:")`
- Line 56: `print(f"[!!!] Continuo con gli step successivi...\n")`
- Line 166: `print(f"\nTutti gli step sono stati eseguiti (vedi sopra per eventuali step falliti). "`

---

### `datasets/Nutrition5k/ExtractFoodInformation.py`
**Italian Docstrings to Translate to English:**
- Line 36 (`normalize_path`): `"Normalizza i caratteri Unicode nel percorso (NFC) per evitare
mismatch di encoding tra il filesystem..."`
- Line 44 (`save_json_safely`): `"Funzione atomica per salvare dati JSON in modo sicuro...."`
- Line 208 (`process_single_image`): `"Processa una singola immagine in modo sequenziale con logging di debug dettagliato...."`

**Italian Comments to Translate (13 found, sampling lines):**
- Line 27: `# percorsi`
- Line 34: `# --- LOGICA DI ESECUZIONE ---`
- Line 215: `# Ricostruisce il path assoluto unendo la base dir al path relativo salvato nel JSON.`
- Line 291: `# --- FASE DI PARSING ---`
- Line 293: `# 1. RIMOSSA LA PARTE DI THINKING (solo se esiste, come in Gemma)`
- Line 295: `# Divide la stringa in due parti usando "<channel|>" e prende l'ultima parte (il JSON)`
- Line 363: `# Solo Gemma 4 richiede il token esplicito per il thinking`
- Line 374: `# Attiviamo la modalità Deep Thinking come default per questo task complesso`
- Line 1202: `# Generazione lazy delle immagini annotate se non esistono ancora fisicamente su disco`
- Line 1217: `# --- CARICAMENTO DATI E FILTRAGGIO ---`
- *(and 3 more Italian comments in this file)*

**Italian Print Statements to Translate (13 found):**
- Line 52: `print(f"[ERRORE] Impossibile salvare il file JSON: {e}")`
- Line 97: `print(f"[YOLO ORIGINALE] Ingredienti rilevati per {os.path.basename(full_image_path)}: {[b['class_name'] for b in raw_boxes]}")`
- Line 250: `print(f"[ERRORE YOLO] {e}")`
- Line 266: `print(f"[ERRORE CRITICO] Il client Ollama è crashato durante ask(): {e}")`
- Line 281: `print(f"[DEBUG] -> Il risultato è esattamente None.")`
- Line 285: `print(f"[DEBUG] -> Il risultato è una stringa vuota.")`
- Line 315: `print(f"[Errore di Parsing] Impossibile analizzare l'output JSON.")`
- Line 316: `print(f"Dettagli errore Pydantic: {e}")`
- Line 381: `print(f"[*] Modello selezionato: {model}")`
- Line 1207: `print(f"[*] Generazione immagine annotata mancante per l'esempio {i+1}...")`
- *(and 3 more Italian prints in this file)*

---

### `datasets/Nutrition5k/ExtractFoodInformationTwoStep.py`
> [!WARNING]
> File has git conflict markers (`<<<<<<<` / `>>>>>>>`). Must be resolved before AST parsing or execution.

**LLM / Dev Residue Comments to Clean:**
- Line 228: `# Salta il passaggio al LLM per il momento`

**Italian Comments to Translate (11 found, sampling lines):**
- Line 26: `# percorsi`
- Line 38: `# --- SCHEMI PYDANTIC PER STEP 1 ---`
- Line 55: `# --- LOGICA DI ESECUZIONE ---`
- Line 141: `# Raggruppa i box che si sovrappongono e calcola la media pesata (WBF a gruppi)`
- Line 195: `# Ordiniamo i box filtrati per confidenza decrescente`
- Line 214: `# Creiamo la label solo con ID incrementale, senza confidenza`
- Line 216: `# Usiamo la funzione box_label che si occupa anche di assegnare i colori YOLO`
- Line 221: `# Mostra l'immagine (premi un tasto per continuare)`
- Line 231: `# STEP 1: Analisi Visiva (Solo pesi e ingredienti)`
- Line 270: `# Passiamo l'immagine per robustezza, come suggerito nel paper`
- *(and 1 more Italian comments in this file)*

**Italian Print Statements to Translate (10 found):**
- Line 68: `print(f"[ERRORE] Impossibile salvare il file JSON: {e}")`
- Line 95: `print(f"[Errore di Parsing] Dettagli errore Pydantic: {e}")`
- Line 138: `print(f"[YOLO ORIGINALE] Ingredienti rilevati: {[b['class_name'] for b in raw_boxes]}")`
- Line 227: `print(f"[ERRORE YOLO] {e}")`
- Line 253: `print(f"[ERRORE] Fallito il parsing dello Step 1.")`
- Line 284: `print(f"[ERRORE] Fallito il parsing dello Step 2.")`
- Line 361: `print(f"[*] Modello selezionato: {model}")`
- Line 458: `print(f"\nAvvio loop di inferenza. {total_to_process} immagini rimanenti da processare.")`
- Line 476: `print("\n[!] Esecuzione interrotta. Salvataggio di emergenza...")`
- Line 481: `print(f"Salvataggio finale completato. Predizioni salvate: {len(existing_predictions)}")`

---

### `datasets/Nutrition5k/ExtractFoodInformationTwoStepHybrid.py`
**LLM / Dev Residue Comments to Clean:**
- Line 225: `# Salta il passaggio al LLM per il momento`

**Italian Comments to Translate (10 found, sampling lines):**
- Line 28: `# percorsi`
- Line 35: `# --- SCHEMI PYDANTIC PER STEP 1 ---`
- Line 52: `# --- LOGICA DI ESECUZIONE ---`
- Line 138: `# Raggruppa i box che si sovrappongono e calcola la media pesata (WBF a gruppi)`
- Line 192: `# Ordiniamo i box filtrati per confidenza decrescente`
- Line 211: `# Creiamo la label solo con ID incrementale, senza confidenza`
- Line 213: `# Usiamo la funzione box_label che si occupa anche di assegnare i colori YOLO`
- Line 218: `# Mostra l'immagine (premi un tasto per continuare)`
- Line 228: `# STEP 1: Analisi Visiva (Solo pesi e ingredienti)`
- Line 1291: `# --- CARICAMENTO DATI ---`

**Italian Print Statements to Translate (10 found):**
- Line 65: `print(f"[ERRORE] Impossibile salvare il file JSON: {e}")`
- Line 92: `print(f"[Errore di Parsing] Dettagli errore Pydantic: {e}")`
- Line 135: `print(f"[YOLO ORIGINALE] Ingredienti rilevati: {[b['class_name'] for b in raw_boxes]}")`
- Line 224: `print(f"[ERRORE YOLO] {e}")`
- Line 254: `print(f"[ERRORE] Fallito il parsing dello Step 1.")`
- Line 293: `print(f"[ERRORE] Fallito il parsing dello Step 2.")`
- Line 386: `print(f"[*] Modello selezionato: {model}")`
- Line 1319: `print(f"\nAvvio loop di inferenza. {total_to_process} immagini rimanenti da processare.")`
- Line 1337: `print("\n[!] Esecuzione interrotta. Salvataggio di emergenza...")`
- Line 1342: `print(f"Salvataggio finale completato. Predizioni salvate: {len(existing_predictions)}")`

---

### `datasets/Nutrition5k/Nutrition5k.py`
**Italian Print Statements to Translate (1 found):**
- Line 121: `print(f"WARNING: immagine non trovata su disco per {dish_id}, record scartato.")`

---

### `datasets/Nutrition5k/count_entries.py`
**Italian Print Statements to Translate (3 found):**
- Line 90: `print("Caricamento del modello SentenceTransformer e calcolo embedding vocabolario...")`
- Line 94: `print(f"Vocabolario caricato con {len(vocabulary)} ingredienti.\n")`
- Line 113: `print(f"{file_name}: {entries} entries | No exact: {no_exact} | Pred total: {pred_total} | Exact: 0/0 (N/A) | Std: 0/0 (N/A)")`

---

### `datasets/Nutrition5k/eval/data_utils.py`
**Italian Docstrings to Translate to English:**
- Line 1 (`Module`): `"data_utils.py
=============
Caricamento vocabolario ingredienti e parsing dei nomi file/modello...."`
- Line 57 (`get_dynamic_variants_and_colors`): `"Scopre dinamicamente le varianti presenti sul disco e assegna un colore...."`

**Italian Comments to Translate (1 found, sampling lines):**
- Line 84: `#a3e0a3', '#47d147', '#1f7a1f'],`

---

### `datasets/Nutrition5k/eval/dot_plot.py`
**Italian Docstrings to Translate to English:**
- Line 1 (`Module`): `"dot_plot.py
===========
Generatore GENERICO del grafico "dot-plot few-shot": è la funzione riusata
d..."`
- Line 16 (`plot_dot_grid`): `"Disegna una griglia di dot-plot (uno per 'field') + i singoli pannelli.

value_fn(group, example, st..."`

**Italian Comments to Translate (2 found, sampling lines):**
- Line 65: `# Linee di connessione: con linestyles usa stile per-strategia; altrimenti`
- Line 66: `# usa il comportamento originale (tutte solide) se connect_lines=True.`

---

### `datasets/Nutrition5k/eval/evaluator.py`
**Italian Docstrings to Translate to English:**
- Line 1 (`Module`): `"evaluator.py
============
Classe di valutazione (NLP + macronutrienti) di una singola run di predizi..."`

---

### `datasets/Nutrition5k/eval/plots.py`
**Italian Docstrings to Translate to English:**
- Line 1 (`Module`): `"plots.py
========
Tutti i grafici "few-shot style" del progetto (NLP, macronutrienti, confronto
stra..."`
- Line 259 (`plot_yolo_optuna_metrics`): `"results[group][f'{n} Example ({strategia})'] = {'macro': {subset: {...}}}
strategies:        elenco ..."`
- Line 389 (`plot_combined_strategies`): `"Genera il grafico combinato.
data_dict: dizionario con chiavi formato da (Modello_Quantizzazione) es..."`

**Italian Comments to Translate (12 found, sampling lines):**
- Line 23: `# 0) Metriche NLP per-modello (bar chart), richiesto esplicitamente in`
- Line 135: `# I singoli F1/IoU sono già generati dalla chiamata combinata sopra;`
- Line 136: `# generiamo qui solo i pannelli singoli per P ed R (non nel combinato).`
- Line 204: `# 3) Confronto strategie Optuna`
- Line 251: `# in strategy_legend_handles usano direttamente la chiave; sostituiamo qui`
- Line 252: `# sotto con etichette leggibili rigenerando solo la legenda non è necessario`
- Line 253: `# perché value_fn/keys coincidono con OPTUNA_STRATEGY_LABELS keys.`
- Line 300: `# Se le strategie includono varianti Optuna, usiamo linestyles per distinguerle`
- Line 335: `# 5) Studio soglia (stile di riferimento da preservare)`
- Line 409: `#fdd0a2"},  # Arancio sbiadito`
- *(and 2 more Italian comments in this file)*

---

### `datasets/Nutrition5k/eval/resources.py`
**Italian Docstrings to Translate to English:**
- Line 1 (`Module`): `"resources.py
=============
Contenitore "pigro" (lazy) per vocabolario / modello NLP / embeddings.

I..."`

---

### `datasets/Nutrition5k/eval/runners.py`
**Italian Docstrings to Translate to English:**
- Line 1 (`Module`): `"runners.py
==========
Orchestrazione delle valutazioni: legge le predizioni, calcola le metriche
(tr..."`
- Line 27 (`_load_optuna_scalars_for_model`): `"Load per-model Optuna scalars and match them to a prediction file...."`
- Line 727 (`evaluate_all_strategies_combined`): `"Legge i CSV prodotti da evaluate_few_shot_learning (normal e RAG) e genera
8 grafici combinati — ide..."`
- Line 837 (`evaluate_all_strategies_combined_extended`): `"Versione estesa di evaluate_all_strategies_combined: include anche Two-Step
e Two-Step Matching (RAG..."`

**Italian Comments to Translate (14 found, sampling lines):**
- Line 46: `# Estrazione pulita (rimuoviamo tutto per avere il nome base pulito)`
- Line 53: `# Per YOLO, il tuo JSON ha il suffisso _tb`
- Line 57: `# Per Standard, il JSON di solito NON ha il suffisso _tb`
- Line 69: `# Ricerca chiave di fallback`
- Line 319: `# Cache locale dei summary Optuna per questo folder`
- Line 326: `# Per Standard escludiamo anche i file hybrid/twostep`
- Line 543: `# 4) Studio soglia (loop di threshold su tutti i modelli one-example)`
- Line 591: `# 5) Valutazione per-modello (one_example): alimenta il plot delle metriche`
- Line 592: `#    NLP (bar chart) e le tabelle riassuntive. Usa la stessa cache CSV`
- Line 593: `#    dell'originale (nlp_evaluation.csv): se un modello è già presente non`
- *(and 4 more Italian comments in this file)*

**Italian Print Statements to Translate (12 found):**
- Line 43: `print(f"[DEBUG] Errore di lettura JSON in {summary_path}: {e}")`
- Line 75: `print(f"[DEBUG] Trovata chiave di fallback {fallback_key} ma mancano gli 'optimal_scalars'")`
- Line 316: `print(f"[!] Cartella predizioni non trovata: {pred_folder}")`
- Line 348: `print(f"[!] Errore YOLO base su {pred_path}: {e}")`
- Line 357: `print(f"[!] Scalars YOLO non trovati per {filename} / {strategy} in {summary_path}")`
- Line 367: `print(f"[!] Errore YOLO {strategy} su {pred_path}: {e}")`
- Line 380: `print(f"[!] Errore Standard base su {pred_path}: {e}")`
- Line 389: `print(f"[!] Scalars Standard non trovati per {filename} / {strategy} in {summary_path}")`
- Line 399: `print(f"[!] Errore Standard {strategy} su {pred_path}: {e}")`
- Line 799: `print("[AVVISO] Nessun dato valido trovato per generare i grafici combinati.")`
- *(and 2 more Italian prints in this file)*

---

### `datasets/Nutrition5k/eval/style.py`
**Italian Docstrings to Translate to English:**
- Line 1 (`Module`): `"style.py
========
Stile grafico UNICO, condiviso da tutti i grafici del pacchetto.

Lo stile di rife..."`
- Line 165 (`style_dot_axis`): `"Cosmesi comune di un pannello dot-plot: griglia, xticks=GROUPS, limiti...."`
- Line 191 (`strategy_legend_handles`): `"Costruisce gli handle di legenda 'marker vuoto colorato', stile few-shot...."`

**Italian Comments to Translate (10 found, sampling lines):**
- Line 18: `# Gruppi di modelli (asse X comune a tutti i grafici)`
- Line 23: `# Palette "strategia" (colore + marker), identica a quella usata nei grafici`
- Line 46: `# stesso simbolo di YOLO, si distingue per linea tratteggiata`
- Line 47: `# stesso simbolo di Standard, si distingue per linea tratteggiata`
- Line 54: `# Stile della linea di connessione per strategia.`
- Line 55: `# Le varianti Optuna usano linea tratteggiata; le strategie base usano linea continua.`
- Line 70: `# Fattore moltiplicativo sulla dimensione scatter (s) per marker che`
- Line 73: `# stella: usata in OPTUNA_STRATEGY_MARKERS per linear_bias_all5`
- Line 76: `# Palette "strategia Optuna" (usata nel confronto tra strategie di scaling)`
- Line 129: `# Limiti/tick specifici per il MAE% in scala log (stile originale)`

---

### `datasets/Nutrition5k/eval/tables.py`
**Italian Docstrings to Translate to English:**
- Line 1 (`Module`): `"tables.py
=========
Tabelle riassuntive stampate a terminale durante la valutazione...."`
- Line 10 (`print_summary_tables`): `"Tabella per-modello (gruppo hardware x strategia), come nello script originale...."`
- Line 79 (`print_optuna_strategy_tables`): `"Una tabella per ogni strategia Optuna con 4 colonne:
Standard | Standard Optuna | YOLO | YOLO Optuna..."`
- Line 146 (`print_few_shot_summary_tables`): `"Tabella few-shot (NLP + Macro overall/exact_match), come nello script originale...."`
- Line 195 (`print_prompting_technique_comparison_table`): `"Tabella unica per Standard, Standard Optuna, YOLO e YOLO Optuna.

Il formato segue print_few_shot_su..."`

**Italian Print Statements to Translate (2 found):**
- Line 168: `print(f"{g:<12} | {v:<25} | {'N/A':<12} | {'N/A':<10} | {'N/A':<13} | {'N/A':<10}")`
- Line 191: `print(f"{g:<12} | {v:<25} | {'N/A':<18} | {'N/A':<18} | {'N/A':<18} | {'N/A':<18} | {'N/A':<18}")`

---

### `datasets/Nutrition5k/finetuning_dataset_extraction.py`
**LLM / Dev Residue Comments to Clean:**
- Line 62: `# Rimosso il filtro: if ing["weight_grams"] < 1.0: continue`

**Italian Comments to Translate (2 found, sampling lines):**
- Line 6: `# Aggiungo la root del progetto al sys.path per importare answer_schema`
- Line 137: `# Esecuzione in locale:`

**Italian Print Statements to Translate (2 found):**
- Line 134: `print(f"Creati {len(train_samples)} esempi di train in {jsonl_output_path}")`
- Line 135: `print(f"Creati {len(val_samples)} esempi di validation in {jsonl_val_output_path}")`

---

### `datasets/Nutrition5k/inspect_dish.py`
**LLM / Dev Residue Comments to Clean:**
- Line 12: `# salta la standardizzazione (più veloce)`

**Italian Docstrings to Translate to English:**
- Line 1 (`Module`): `"inspect_dish.py
===============
Dato un food_id, stampa per ogni file JSON in data/prediction/five_e..."`
- Line 67 (`standardize`): `"Restituisce la lista di ingredienti standardizzati (uno per uno, con indicazione)...."`
- Line 91 (`_get_name`): `"Estrae il nome da un dizionario ingrediente (gestisce formati diversi)...."`
- Line 100 (`extract_predicted_names`): `"Restituisce i nomi raw degli ingredienti predetti (visible + hidden).

Gestisce i diversi formati:
-..."`
- Line 132 (`extract_gt_names`): `"Restituisce i nomi raw degli ingredienti della ground truth...."`
- Line 143 (`model_label`): `"Converte il nome file in un'etichetta compatta per la tabella...."`
- Line 179 (`_wrap`): `"Spezza una stringa in righe di al max `width` caratteri...."`
- Line 186 (`print_table`): `"rows: lista di dict con chiavi:
    label        str   etichetta modello
    pred_raw     list  ingr..."`
- Line 226 (`multiline_row`): `"Stampa una riga che può espandersi su più righe per gli ingredienti...."`

**Italian Comments to Translate (14 found, sampling lines):**
- Line 50: `# Caricamento vocabolario + modello NLP (solo se serve)`
- Line 88: `# Helper: parsing ingredienti da un entry JSON`
- Line 109: `# RAG: ha la precedenza su tutto`
- Line 115: `# Cerca la chiave giusta in ordine di preferenza`
- Line 149: `# Tecnica di prompting`
- Line 161: `# Modello hardware`
- Line 176: `# Stampa tabella`
- Line 194: `# colonna ingredienti raw / std`
- Line 259: `# ---- Predizioni modelli ----`
- Line 303: `# Assicura che lo script funzioni sia lanciato dalla root del progetto`
- *(and 4 more Italian comments in this file)*

**Italian Print Statements to Translate (8 found):**
- Line 317: `print(f"[ERRORE] Directory predizioni non trovata: {pred_dir}")`
- Line 326: `print(f"[ERRORE] Nessun file JSON trovato in: {pred_dir}")`
- Line 335: `print(f"[INFO] Caricamento vocabolario da {vocab_csv}...")`
- Line 338: `print("[INFO] Caricamento modello NLP (all-MiniLM-L6-v2)...")`
- Line 342: `print("[INFO] Calcolo embeddings vocabolario...")`
- Line 356: `print(f"[WARN] {fname}: formato inatteso (non è una lista), saltato.")`
- Line 388: `print(f"\n[ERRORE] Il food_id '{food_id}' non è stato trovato in nessun file JSON di '{pred_dir}'.")`
- Line 389: `print("Suggerimento: verifica l'ID con --list (oppure controlla i file manualmente).")`

---

### `datasets/Nutrition5k/optimization/run_optuna.py`
**LLM / Dev Residue Comments to Clean:**
- Line 216: `# NUOVO: calcola il range di soglia specifico per questo nutriente`

**Italian Docstrings to Translate to English:**
- Line 71 (`get_gt_val`): `"Estrae il valore ground truth per un dato nutriente...."`
- Line 87 (`get_pred_val`): `"Estrae il valore predetto per un dato nutriente...."`
- Line 183 (`get_pred_range`): `"Calcola il range [min, max] dei valori predetti per un dato nutriente
nel train set, con un margine ..."`
- Line 208 (`optimize_piecewise_all5`): `"Strategia: A soglia (Piecewise). Scale diverso se pred_val è minore o maggiore di un limite...."`
- Line 256 (`optimize_linear_bias_all5`): `"Strategia: Modello lineare con bias. adj_val = pred_val * scale + bias..."`
- Line 320 (`optimize_for_file`): `"Entry point: ottimizza un file di predizioni con la strategia indicata...."`
- Line 333 (`run_strategy`): `"Esegue una strategia completa su tutti i json_files e salva il summary.
Ogni studio Optuna usa n_job..."`

**Italian Comments to Translate (26 found, sampling lines):**
- Line 8: `# 1. PERCORSI`
- Line 14: `# Imposta qui la strategia da usare, oppure tramite variabile d'ambiente OPTUNA_STRATEGY.`
- Line 17: `#   "linear_mass_only" -> ottimizza solo la massa (1 parametro)`
- Line 19: `#   "piecewise_all5"   -> correzione a soglia per valori piccoli/grandi (15 parametri)`
- Line 20: `#   "linear_bias_all5" -> modello lineare con scale e bias (10 parametri)`
- Line 65: `# Carichiamo la ground truth una volta sola all'avvio`
- Line 152: `# Ampio range per la potenza`
- Line 166: `# Applica la formula non lineare (evita crash se pred_val è esattamente 0)`
- Line 191: `# scarta valori nulli/non validi`
- Line 194: `# fallback di sicurezza se non ci sono valori validi`
- *(and 16 more Italian comments in this file)*

**Italian Print Statements to Translate (9 found):**
- Line 66: `print("Caricamento Ground Truth...")`
- Line 380: `print(f"{prefix} [!] Errore su {filepath}: {e}")`
- Line 430: `print(f"[ERRORE] RUN_MODE='{RUN_MODE}' non valido. Scegli tra: both, rag, nonrag")`
- Line 443: `print("Modalità: solo cartelle RAG (*_rag)")`
- Line 450: `print("Modalità: solo cartelle non-RAG")`
- Line 453: `print("Modalità: tutte le cartelle (RAG + non-RAG)")`
- Line 464: `print("Nessun file JSON trovato. Controlla il percorso o il pattern di ricerca.")`
- Line 476: `print(f"File di predizione trovati: {len(json_files)}")`
- Line 487: `print("✅ Ottimizzazione batch completata per tutte le strategie!")`

---

### `datasets/Nutrition5k/optimization/sample_optuna_subset.py`
**Italian Comments to Translate (9 found, sampling lines):**
- Line 9: `# Percorsi dei file`
- Line 16: `# 1. Estrarre gli ID dei dish utilizzati come few-shot example dal codice sorgente`
- Line 20: `# Cerchiamo tutti i pattern tipo "dish_1562012076"`
- Line 26: `# 2. Caricare il dataset di training`
- Line 42: `# 4. Sostituire i dish utilizzati come esempi se sono stati estratti`
- Line 43: `# Creiamo un set degli identificativi estratti per escluderli dal pool`
- Line 46: `# Prepariamo un pool di elementi validi per le sostituzioni (non esclusi e non già estratti)`
- Line 61: `# Sostituiamo gli elementi non validi in sampled_data`
- Line 86: `# 5. Salvataggio del nuovo dataset`

**Italian Print Statements to Translate (4 found):**
- Line 27: `print(f"[*] Caricamento del dataset di training da: {TRAIN_DATASET_PATH}")`
- Line 37: `print(f"[!] Attenzione: i dati originali ({len(train_data)}) sono minori del campione richiesto ({sample_size}).")`
- Line 83: `print(f"[*] Sostituiti {replace_count} dish di esempio che erano stati estratti.")`
- Line 90: `print(f"[*] Subset estratto salvato con successo in: {OUTPUT_PATH}")`

---

### `datasets/Nutrition5k/perfect_prediction.py`
**Italian Print Statements to Translate (1 found):**
- Line 35: `print(f"File '{output_file}' creato con successo! Contiene {len(perfect_predictions)} predizioni perfette.")`

---

### `datasets/Nutrition5k/run_yolo_annotate.py`
**Italian Docstrings to Translate to English:**
- Line 1 (`Module`): `"run_yolo_annotate.py
====================
Esegue YOLO sui piatti specificati (come in ExtractFoodInf..."`
- Line 63 (`run_yolo_on_image`): `"Identica a ExtractFoodInformation.run_yolo_on_image, ma riceve
i pesi come parametro esplicito invec..."`

**Italian Comments to Translate (3 found, sampling lines):**
- Line 32: `# Percorsi fissi (stessi di ExtractFoodInformation.py)`
- Line 103: `# --- Filtraggio NMS per IoU > 0.5 tra box della stessa classe ---`
- Line 184: `# --- Salvataggio ---`

**Italian Print Statements to Translate (4 found):**
- Line 72: `print(f"[INFO] Caricamento modello YOLO: {weights_path}")`
- Line 208: `print(f"[ERRORE] Pesi YOLO non trovati: {weights_path}")`
- Line 223: `print(f"[PIATTO] {dish_id}")`
- Line 227: `print(f"[ERRORE] Immagine non trovata: {full_path}")`

---

### `patch_draw.py`
**Italian Comments to Translate (5 found, sampling lines):**
- Line 9: `# Linea di Baseline e Testo inline con Bounding Box`
- Line 14: `# Bounding Box per l'etichetta, posizionata a sinistra`
- Line 21: `# Linee divisorie verticali chiare tra i gruppi di modelli`
- Line 26: `# Nessun dodging: la posizione X dipende solo dal numero di esempi`
- Line 45: `# Uso alpha=0.85 per gestire l'overplotting senza spostare i punti`

---

### `revert_draw.py`
**Italian Comments to Translate (1 found, sampling lines):**
- Line 18: `# Linee divisorie verticali chiare tra i gruppi di modelli`

---

### `settings.py`
**Italian Docstrings to Translate to English:**
- Line 1 (`Module`): `"Configurazione centralizzata del progetto LLMforFood.

Questo modulo è l'unico punto in cui viene ca..."`
- Line 36 (`MissingEnvVarError`): `"Sollevata quando una variabile obbligatoria non è definita nel ``.env``...."`
- Line 40 (`get_env`): `"Restituisce il valore di una variabile d'ambiente definita nel ``.env``.

Args:
    name: Nome della..."`
- Line 70 (`get_path`): `"Come :func:`get_env`, ma espande ``~`` e restituisce un percorso assoluto...."`

**Italian Comments to Translate (5 found, sampling lines):**
- Line 24: `#: Directory radice del progetto (quella che contiene ``.env``).`
- Line 27: `#: Percorso del file ``.env`` caricato all'import di questo modulo.`
- Line 30: `# Il caricamento usa un percorso assoluto: funziona indipendentemente dalla`
- Line 31: `# working directory da cui viene lanciato lo script. Le variabili già presenti`
- Line 32: `# nell'ambiente di shell hanno la precedenza (override=False).`

---

