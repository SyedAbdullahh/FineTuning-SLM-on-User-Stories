from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover - allow environments without torch for dry-run
    torch = None
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .config import CONFIG, ensure_output_dir


def tokenize_batch(batch, tokenizer, max_length: int):
    return tokenizer(batch["text"], truncation=True, max_length=max_length)


class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(features, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # If torch is unavailable or a dry-run is requested, skip loading the model
    if torch is None:
        return None, tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model, tokenizer


def build_lora_model(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG.lora_rank,
        lora_alpha=CONFIG.lora_alpha,
        lora_dropout=CONFIG.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with LoRA for user story generation")
    parser.add_argument("--model-name", type=str, default=CONFIG.default_model_name)
    parser.add_argument("--train-file", type=Path, default=CONFIG.output_dir / "train.jsonl")
    parser.add_argument("--validation-file", type=Path, default=CONFIG.output_dir / "validation.jsonl")
    parser.add_argument("--output-dir", type=Path, default=CONFIG.output_dir / "lora_model")
    parser.add_argument("--max-length", type=int, default=CONFIG.max_length)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--dry-run", action="store_true", help="Run a lightweight dry run without loading the model")
    args = parser.parse_args()
    # Lightweight dry-run: only load and tokenize dataset, then exit
    raw_datasets = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.validation_file)},
    )

    if args.dry_run:
        print("Dry run: tokenizing and reporting dataset sizes")
        # load tokenizer only for tokenization
        _, tokenizer = load_model_and_tokenizer(args.model_name)
        tokenized_train = raw_datasets["train"].map(
            lambda batch: tokenize_batch(batch, tokenizer, args.max_length),
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )
        print("Train examples:", len(tokenized_train))
        tokenized_val = raw_datasets["validation"].map(
            lambda batch: tokenize_batch(batch, tokenizer, args.max_length),
            batched=True,
            remove_columns=raw_datasets["validation"].column_names,
        )
        print("Validation examples:", len(tokenized_val))
        print("Dry run complete. Add --dry-run to skip heavy model operations.")
        return

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model = build_lora_model(model)

    tokenized = raw_datasets.map(
        lambda batch: tokenize_batch(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
        bf16=False,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=CausalLMCollator(tokenizer),
    )

    trainer.train()
    ensure_output_dir()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
