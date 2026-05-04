from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .config import CONFIG, SYSTEM_PROMPT, ensure_output_dir


def build_prompt(requirement: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\nConvert the following software requirement into a user story.\n"
        f"Requirement: {requirement}\n"
        f"<|assistant|>\n"
    )


def load_dataset(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["samples"]


def split_samples(
    samples: list[dict[str, Any]],
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be less than 1.0")

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * validation_ratio)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def enrich_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for sample in samples:
        enriched.append(
            {
                "id": sample["id"],
                "domain": sample["domain"],
                "input": sample["input"],
                "output": sample["output"],
                "prompt": build_prompt(sample["input"]),
                "text": build_prompt(sample["input"]) + sample["output"],
            }
        )
    return enriched


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset splits for user story fine-tuning")
    parser.add_argument("--dataset", type=Path, default=CONFIG.dataset_path)
    parser.add_argument("--output-dir", type=Path, default=CONFIG.output_dir)
    parser.add_argument("--train-ratio", type=float, default=CONFIG.train_ratio)
    parser.add_argument("--validation-ratio", type=float, default=CONFIG.validation_ratio)
    parser.add_argument("--seed", type=int, default=CONFIG.split_seed)
    args = parser.parse_args()

    samples = load_dataset(args.dataset)
    train, validation, test = split_samples(samples, args.train_ratio, args.validation_ratio, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared_train = enrich_samples(train)
    prepared_validation = enrich_samples(validation)
    prepared_test = enrich_samples(test)

    write_jsonl(args.output_dir / "train.jsonl", prepared_train)
    write_jsonl(args.output_dir / "validation.jsonl", prepared_validation)
    write_jsonl(args.output_dir / "test.jsonl", prepared_test)

    summary = {
        "source_dataset": str(args.dataset),
        "train_samples": len(prepared_train),
        "validation_samples": len(prepared_validation),
        "test_samples": len(prepared_test),
        "prompt_format": "<|system|> / <|user|> / <|assistant|>",
    }
    (args.output_dir / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
