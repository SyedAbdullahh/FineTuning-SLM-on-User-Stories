# SLM Fine-Tuning for User Stories

This project fine-tunes a small Qwen model with LoRA to generate concise, industry-compliant user stories from software requirements.

## Current status

- Dataset generation: complete
- Remaining execution plan: in progress

## Suggested model

Default base model:

- `Qwen/Qwen2.5-0.5B-Instruct`

The model can be changed through environment variables or script arguments.

## Project flow

1. Preprocess the dataset and create train / validation / test splits
2. Fine-tune the Qwen base model with LoRA
3. Compare baseline and fine-tuned outputs
4. Save evaluation results and example generations

## Files

- `user_story_dataset.json` - curated dataset
- `src/data_prep.py` - dataset preprocessing and splitting
- `src/train_lora.py` - LoRA fine-tuning pipeline
- `src/evaluate.py` - baseline vs fine-tuned evaluation
- `src/config.py` - shared configuration

## Workflow

### 1) Prepare data

Run the preprocessing script to create JSONL splits in `outputs/`.

```bash
python -m src.data_prep --dataset user_story_dataset.json --output-dir outputs
```

### 2) Train

Run the full fine-tuning wrapper. It preprocesses the dataset and then starts LoRA training.

```bash
python scripts/finetune_qwen_lora.py --dataset user_story_dataset.json --output-dir outputs/lora_model
```

The training setup masks prompt tokens and learns only from the user-story completion, which is the intended supervised fine-tuning behavior.

### 3) Evaluate

Run the evaluation script to produce a metrics report and sample comparisons.

```bash
python -m src.evaluate --test-file outputs/test.jsonl --output-file outputs/evaluation_report.json
```

### Generate baseline vs fine-tuned samples

Create a JSONL with side-by-side generations to show differences:

```bash
PYTHONPATH=. python scripts/generate_comparisons.py --test-file outputs/test.jsonl --output-file outputs/comparisons.jsonl --limit 50
```

Open `outputs/comparisons.jsonl` to inspect baseline vs fine-tuned outputs.

### 4) Interactive CLI

After fine-tuning, start an interactive prompt loop:

```bash
python scripts/chat_qwen_cli.py --adapter-path outputs/lora_model
```

### Quick dry-run

Use the provided script to run the pipeline in dry-run mode (no heavy downloads):

```bash
./scripts/run_pipeline.sh
```

For full training and evaluation, run the Python modules directly and omit `--dry-run`. Ensure you have sufficient disk, network, and (optionally) GPU resources.

## Notes

- The scripts are written to be configurable and lightweight.
- The evaluation stage includes format compliance checks and overlap-based metrics.
- The training script is designed for Hugging Face Transformers + PEFT.
- For full training, install dependencies from `requirements.txt` first.
- The fine-tuning wrapper validates the dataset path and the CLI warns if the adapter folder is missing.
