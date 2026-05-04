from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    dataset_path: Path = Path("user_story_dataset.json")
    output_dir: Path = Path("outputs")
    split_seed: int = 42
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    max_length: int = 256
    default_model_name: str = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


CONFIG = ProjectConfig()


SYSTEM_PROMPT = (
    "You are a software engineering assistant that converts requirements into "
    "short, clear, Agile-compliant user stories."
)


def ensure_output_dir() -> Path:
    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
    return CONFIG.output_dir
