from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import CONFIG


FORMAT_PATTERN = re.compile(r"^As an? .+?, I want to .+? so that .+?\.$", re.IGNORECASE)


def is_format_compliant(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    return bool(FORMAT_PATTERN.match(normalized))


def simple_overlap(reference: str, candidate: str) -> float:
    ref_tokens = set(reference.lower().split())
    cand_tokens = set(candidate.lower().split())
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & cand_tokens) / len(ref_tokens)


def extract_completion(text: str) -> str:
    marker = "<|assistant|>"
    if marker in text:
        return text.split(marker, maxsplit=1)[-1].strip()
    return text.strip()


def load_base_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(model_name: str, adapter_path: Path):
    model, tokenizer = load_base_model(model_name)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return extract_completion(tokenizer.decode(output_ids[0], skip_special_tokens=False))


def dataset_to_records(dataset):
    return [dataset[index] for index in range(len(dataset))]


def evaluate_model(model, tokenizer, records, limit: int | None = None):
    used_records = records if limit is None else records[:limit]
    outputs = []
    compliant = 0
    overlap_sum = 0.0

    for record in used_records:
        prompt = record["prompt"]
        reference = record["output"]
        generated = generate(model, tokenizer, prompt)
        outputs.append(
            {
                "id": record["id"],
                "reference": reference,
                "generated": generated,
                "format_compliant": is_format_compliant(generated),
                "overlap": simple_overlap(reference, generated),
            }
        )
        if outputs[-1]["format_compliant"]:
            compliant += 1
        overlap_sum += outputs[-1]["overlap"]

    total = len(outputs) or 1
    return {
        "format_compliance_rate": compliant / total,
        "average_overlap": overlap_sum / total,
        "samples_evaluated": total,
        "examples": outputs[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline and fine-tuned user story models")
    parser.add_argument("--model-name", type=str, default=CONFIG.default_model_name)
    parser.add_argument("--adapter-path", type=Path, default=CONFIG.output_dir / "lora_model")
    parser.add_argument("--test-file", type=Path, default=CONFIG.output_dir / "test.jsonl")
    parser.add_argument("--output-file", type=Path, default=CONFIG.output_dir / "evaluation_report.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Run evaluation without loading models; uses references as 'generated' to compute format compliance and overlap")
    args = parser.parse_args()

    test_dataset = load_dataset("json", data_files=str(args.test_file))["train"]
    test_records = dataset_to_records(test_dataset)

    if args.dry_run:
        # In dry-run mode, we use the reference outputs as the generated text.
        print("Dry run evaluation: using reference outputs as generated outputs")
        outputs = []
        compliant = 0
        overlap_sum = 0.0
        used = test_records[: args.limit]
        for record in used:
            reference = record["output"]
            generated = reference
            ok = is_format_compliant(generated)
            overlap = simple_overlap(reference, generated)
            outputs.append({
                "id": record["id"],
                "reference": reference,
                "generated": generated,
                "format_compliant": ok,
                "overlap": overlap,
            })
            if ok:
                compliant += 1
            overlap_sum += overlap

        total = len(outputs) or 1
        baseline_report = {
            "format_compliance_rate": compliant / total,
            "average_overlap": overlap_sum / total,
            "samples_evaluated": total,
            "examples": outputs[:10],
        }
        finetuned_report = baseline_report

    else:
        baseline_model, baseline_tokenizer = load_base_model(args.model_name)
        fine_tuned_model, fine_tuned_tokenizer = load_finetuned_model(args.model_name, args.adapter_path)

        baseline_report = evaluate_model(baseline_model, baseline_tokenizer, test_records, args.limit)
        finetuned_report = evaluate_model(fine_tuned_model, fine_tuned_tokenizer, test_records, args.limit)

    report = {
        "baseline": baseline_report,
        "fine_tuned": finetuned_report,
        "delta": {
            "format_compliance_rate": finetuned_report["format_compliance_rate"] - baseline_report["format_compliance_rate"],
            "average_overlap": finetuned_report["average_overlap"] - baseline_report["average_overlap"],
        },
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
