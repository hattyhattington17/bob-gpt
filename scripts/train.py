"""Entry point for training Bob on bobified Shakespeare text.

Usage:
    uv run python scripts/train.py --config configs/nano.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bob.config import ModelConfig, TrainingConfig
from bob.training.trainer import train


def main() -> None:
    """Parse args, detect device, and run the training loop."""
    parser = argparse.ArgumentParser(description="Train Bob on bobified Shakespeare.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    model_config = ModelConfig.from_yaml(config_path)
    training_config = TrainingConfig.from_yaml(config_path)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    train(model_config, training_config, device)


if __name__ == "__main__":
    main()
