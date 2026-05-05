from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src` imports work when running this script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.config import CONFIG
from src.data_prep import enrich_samples, load_dataset, split_samples, write_jsonl
from src.train_lora import main as train_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Qwen small language model with LoRA on the user story dataset"
    )
    parser.add_argument("--dataset", type=Path, default=CONFIG.dataset_path, help="Path to user_story_dataset.json")
    parser.add_argument("--data-dir", type=Path, default=CONFIG.output_dir, help="Directory for generated train/validation/test splits")
    parser.add_argument("--output-dir", type=Path, default=CONFIG.output_dir / "lora_model", help="Directory to save the adapter")
    parser.add_argument("--model-name", type=str, default=CONFIG.default_model_name, help="Base Qwen model name")
    parser.add_argument("--train-ratio", type=float, default=CONFIG.train_ratio, help="Train split ratio")
    parser.add_argument("--validation-ratio", type=float, default=CONFIG.validation_ratio, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=CONFIG.split_seed, help="Random seed for splitting")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=CONFIG.max_length, help="Maximum token length")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    # Step 1: preprocess data so the training run is self-contained.
    samples = load_dataset(args.dataset)
    train, validation, test = split_samples(samples, args.train_ratio, args.validation_ratio, args.seed)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.data_dir / "train.jsonl", enrich_samples(train))
    write_jsonl(args.data_dir / "validation.jsonl", enrich_samples(validation))
    write_jsonl(args.data_dir / "test.jsonl", enrich_samples(test))

    # Step 2: reuse the training implementation already in src/train_lora.py.
    # The wrapper keeps the user-facing command simple.
    import sys

    sys.argv = [
        "train_lora.py",
        "--model-name",
        args.model_name,
        "--train-file",
        str(args.data_dir / "train.jsonl"),
        "--validation-file",
        str(args.data_dir / "validation.jsonl"),
        "--output-dir",
        str(args.output_dir),
        "--max-length",
        str(args.max_length),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
    ]
    train_main()


if __name__ == "__main__":
    main()
