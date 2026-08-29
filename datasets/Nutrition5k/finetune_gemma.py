import os
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import torch
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

def parse_args():
    parser = argparse.ArgumentParser(description="Finetuning Multimodale di Gemma 4 con QLoRA")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-e2b-it",
                        help="HuggingFace model ID")
    parser.add_argument("--train_path", type=str, default="./gemma4_train_data.jsonl",
                        help="Path al file JSONL di training")
    parser.add_argument("--val_path", type=str, default="./gemma4_val_data.jsonl",
                        help="Path al file JSONL di validation")
    parser.add_argument("--output_dir", type=str, default="./gemma4_vision_finetuned")
    parser.add_argument("--epochs", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Configurazione Quantizzazione 4-bit (QLoRA)
    print("Configurando quantizzazione a 4-bit (Q4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Caricamento Modello Multimodale e Processor
    print(f"Caricando il modello vision {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)

    # Capisce in automatico se si trova sulla GPU 0 o sulla GPU 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto"  # Lasciamo che divida il peso al 50% tra le due GPU
    )

    # --- IL BYPASS SALVA-MEMORIA ---
    print("Preparazione manuale del modello (Bypass float32)...")

    # 1. Congela i pesi del modello base
    for param in model.parameters():
        param.requires_grad = False

    # 2. Abilita il gradient checkpointing per risparmiare memoria durante i passaggi
    model.gradient_checkpointing_enable()

    # 3. Forza l'input a richiedere i gradienti (necessario per PEFT)
    model.enable_input_require_grads()
    # -------------------------------

    # 3. Configurazione LoRA
    print("Inizializzazione adapter LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # AGGIUNTO ".linear" a ogni target per bypassare il Gemma4ClippableLinear
        target_modules=[
            "q_proj.linear",
            "k_proj.linear",
            "v_proj.linear",
            "o_proj.linear",
            "gate_proj.linear",
            "up_proj.linear",
            "down_proj.linear"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Caricamento Dataset JSONL
    print("Caricamento dataset di training e validation...")
    train_dataset = load_dataset("json", data_files=args.train_path, split="train")
    val_dataset = load_dataset("json", data_files=args.val_path, split="train")

    # --- AGGIORNAMENTO DINAMICO DEI PATH SU KAGGLE ---
    KAGGLE_IMG_ROOT = "/kaggle/input/datasets/gillesokhin/nutrition5k-dataset/"

    def fix_image_paths(example):
        # Se siamo su Kaggle, anteponi la root del dataset ai path esistenti
        if os.path.exists("/kaggle/input"):
            updated_images = []
            for img_path in example["images"]:
                # Pulisce eventuali slash sballate e unisce i path
                clean_path = img_path.replace("\\", "/")
                if not clean_path.startswith("/kaggle/input"):
                    clean_path = os.path.join(KAGGLE_IMG_ROOT, clean_path)
                updated_images.append(clean_path)
            example["images"] = updated_images
        return example

    print("Aggiornamento dinamico dei path delle immagini...")
    train_dataset = train_dataset.map(fix_image_paths)
    val_dataset = val_dataset.map(fix_images_path if 'fix_images_path' else fix_image_paths)

    # ------------------------------------------------

    # --- INIZIO CORREZIONE FORMATO MESSAGGI ---
    def fix_messages(example):
        for msg in example["messages"]:
            # Se il contenuto è una stringa (es. la risposta dell'assistant),
            # avvolgilo nel dizionario text-only che si aspetta la libreria
            if isinstance(msg["content"], str):
                msg["content"] = [{"type": "text", "text": msg["content"]}]
        return example

    print("Allineamento struttura messaggi...")
    train_dataset = train_dataset.map(fix_messages)
    val_dataset = val_dataset.map(fix_messages)

    # 5. Argomenti di Addestramento per VLM
    # 5. Argomenti di Addestramento Standard
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        logging_steps=5,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=30,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        report_to="none",
        remove_unused_columns=False
        # NOTA: dataset_kwargs è stato rimosso da qui
    )

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        args=training_args
    )

    # 7. Addestramento
    start = time.time()

    print("Avvio del training multimodale...")
    trainer.train()

    stop = time.time()

    print("Tempo di addestramento: ", stop - start, "ms")

    # 8. Estrazione e Salvataggio del Grafico della Loss
    print("Generazione del grafico della Loss...")
    history = trainer.state.log_history
    train_steps = [x["step"] for x in history if "loss" in x]
    train_loss = [x["loss"] for x in history if "loss" in x]
    val_steps = [x["step"] for x in history if "eval_loss" in x]
    val_loss = [x["eval_loss"] for x in history if "eval_loss" in x]

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, label="Training Loss", color="blue", alpha=0.7)
    if val_loss:
        plt.plot(val_steps, val_loss, label="Validation Loss", color="orange", linewidth=2)

    plt.xlabel("Training Steps")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Curva di Addestramento - Gemma 4 su Nutrition5k")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plot_path = os.path.join(args.output_dir, "loss_curve.png")
    plt.savefig(plot_path)
    print(f"Grafico salvato in {plot_path}")

    # 9. Salvataggio Pesi Fusi
    print(f"Salvataggio del modello unito in {args.output_dir}_merged")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(f"{args.output_dir}_merged")
    processor.save_pretrained(f"{args.output_dir}_merged")
    print("Addestramento completato con successo!")


if __name__ == "__main__":
    main()