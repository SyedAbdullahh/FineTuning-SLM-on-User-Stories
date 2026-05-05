from __future__ import annotations

import sys
from pathlib import Path
import argparse

# Ensure local package imports work when running the script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, SYSTEM_PROMPT


def build_prompt(question: str) -> str:
    return (
        "<|system|>\n"
        f"{SYSTEM_PROMPT} Return exactly one complete user story ending with a period.\n"
        "<|user|>\n"
        f"{question}\n"
        "<|assistant|>\n"
    )


def generate_complete_response(model, tokenizer, torch, prompt: str, max_new_tokens: int) -> str:
    completion = ""
    current_prompt = prompt

    for _ in range(3):
        inputs = tokenizer(current_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = out_ids[0][input_len:]
        chunk = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if chunk:
            completion = (completion + " " + chunk).strip()

        if completion.endswith((".", "!", "?")):
            break

        current_prompt = prompt + completion + "\n"

    return completion.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive baseline CLI for the base SLM (no adapter)")
    parser.add_argument("--model-name", type=str, default=CONFIG.default_model_name)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Install transformers and torch to run baseline interactive CLI") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=None if device == "cpu" else "auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    if device == "cpu":
        model.to("cpu")
    model.eval()

    print(f"Baseline CLI ready (model={args.model_name}, device={device}). Type a requirement or 'exit'.")
    while True:
        try:
            req = input("requirement> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not req:
            continue
        if req.lower() in {"exit", "quit"}:
            print("bye")
            break

        prompt = build_prompt(req)
        gen = generate_complete_response(model, tokenizer, torch, prompt, args.max_new_tokens)

        print(f"baseline> {gen}\n")


if __name__ == "__main__":
    main()
