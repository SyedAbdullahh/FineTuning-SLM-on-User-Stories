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

### 2) Train

Run the LoRA training script after the splits are ready. A lightweight `--dry-run` mode is available to validate tokenization and dataset sizing without downloading models.

### 3) Evaluate

Run the evaluation script to produce a metrics report and sample comparisons.

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
