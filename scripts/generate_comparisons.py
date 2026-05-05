from __future__ import annotations

import sys
from pathlib import Path
import argparse
import json

# Ensure local package imports work when running the script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import (
    load_base_model,
    load_finetuned_model,
    is_format_compliant,
    simple_overlap,
    dataset_to_records,
)
from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline vs fine-tuned outputs for comparison")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-path", type=Path, default=Path("outputs/lora_model"))
    parser.add_argument("--test-file", type=Path, default=Path("outputs/test.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("outputs/comparisons.jsonl"))
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    if not args.test_file.exists():
        raise FileNotFoundError(f"Test file not found: {args.test_file}")

    ds = load_dataset("json", data_files=str(args.test_file))["train"]
    records = dataset_to_records(ds)
    records = records[: args.limit]

    base_model, base_tokenizer = load_base_model(args.model_name)
    finetuned_model, finetuned_tokenizer = None, None
    try:
        finetuned_model, finetuned_tokenizer = load_finetuned_model(args.model_name, args.adapter_path)
    except Exception:
        print(f"Warning: could not load adapter from {args.adapter_path}; continuing with baseline only")

    out_path = args.output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            prompt = rec.get("prompt")
            reference = rec.get("output")

            baseline = base_model.generate(base_tokenizer(prompt, return_tensors="pt").input_ids.to(base_model.device), max_new_tokens=64)
            baseline_text = base_tokenizer.decode(baseline[0], skip_special_tokens=True)

            finetuned_text = None
            if finetuned_model is not None:
                fin = finetuned_model.generate(finetuned_tokenizer(prompt, return_tensors="pt").input_ids.to(finetuned_model.device), max_new_tokens=64)
                finetuned_text = finetuned_tokenizer.decode(fin[0], skip_special_tokens=True)

            item = {
                "id": rec.get("id"),
                "input": rec.get("input"),
                "reference": reference,
                "baseline": baseline_text,
                "finetuned": finetuned_text,
                "baseline_format_compliant": is_format_compliant(baseline_text),
                "finetuned_format_compliant": is_format_compliant(finetuned_text) if finetuned_text else None,
                "baseline_overlap": simple_overlap(reference, baseline_text),
                "finetuned_overlap": simple_overlap(reference, finetuned_text) if finetuned_text else None,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote comparisons to {out_path}")


if __name__ == "__main__":
    main()
