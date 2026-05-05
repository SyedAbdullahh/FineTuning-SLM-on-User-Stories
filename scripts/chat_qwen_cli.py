from __future__ import annotations

import utils.session
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src` imports work when running this script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.config import CONFIG


def build_prompt(question: str) -> str:
    return (
        "<|system|>\n"
        "You are a software engineering assistant that converts requirements into short, clear, Agile-compliant user stories.\n"
        "<|user|>\n"
        f"Convert the following requirement into a user story:\n{question}\n"
        "<|assistant|>\n"
    )


def load_model_and_tokenizer(model_name: str, adapter_path: Path | None):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "This script requires torch, transformers, and peft. Install project dependencies first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    if adapter_path and adapter_path.exists():
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer, torch


def generate_response(model, tokenizer, torch_module, prompt: str, max_new_tokens: int = 96) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch_module.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>", 1)[-1].strip()
    return decoded.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interact with the fine-tuned Qwen user-story model in CLI")
    parser.add_argument("--model-name", type=str, default=CONFIG.default_model_name, help="Base model name")
    parser.add_argument("--adapter-path", type=Path, default=CONFIG.output_dir / "lora_model", help="LoRA adapter directory")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Maximum generated tokens")
    args = parser.parse_args()

    if not args.adapter_path.exists():
        print(f"Warning: adapter path not found: {args.adapter_path}. Base model will be used.")

    model, tokenizer, torch_module = load_model_and_tokenizer(args.model_name, args.adapter_path)

    print("User story CLI ready. Type a software requirement, or 'exit' to quit.")
    while True:
        try:
            user_input = input("requirement> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not user_input or user_input.lower() in {"exit", "quit"}:
            print("bye")
            break

        prompt = build_prompt(user_input)
        response = generate_response(model, tokenizer, torch_module, prompt, max_new_tokens=args.max_new_tokens)
        print(f"story> {response}\n")


if __name__ == "__main__":
    main()
