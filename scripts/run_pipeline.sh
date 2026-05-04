#!/usr/bin/env bash
# Lightweight pipeline runner: prepares data and runs dry-run train and evaluation
set -euo pipefail
PYTHON=/usr/bin/python3
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1) prepare splits
$PYTHON -m src.data_prep --dataset user_story_dataset.json --output-dir outputs

# 2) dry-run train (tokenize + stats)
$PYTHON -m src.train_lora --train-file outputs/train.jsonl --validation-file outputs/validation.jsonl --dry-run

# 3) dry-run eval (uses references as generated)
$PYTHON -m src.evaluate --test-file outputs/test.jsonl --dry-run

echo "Pipeline dry-run complete. For full training, remove --dry-run and ensure model and GPU availability."
