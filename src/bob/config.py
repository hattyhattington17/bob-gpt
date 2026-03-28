"""Load and validate model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)  # immutable
class ModelConfig:
    """Immutable configuration for the model architecture."""

    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_seq_len: int
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True

    @property
    def d_head(self) -> int:
        """Return the dimension of each attention head."""
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.d_head % 2 != 0:
            raise ValueError(f"d_head ({self.d_head}) must be even for RoPE")

    @classmethod
    def from_yaml(cls, path: Path) -> ModelConfig:
        """Load model config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw["model"])


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable configuration for the training loop."""

    data_path: str  # path to training data
    train_split: float  # fraction of data for training vs validation
    batch_size: int  # sequences per batch
    max_steps: int  # number of training steps to run the LR schedule over
    warmup_steps: int  # how many steps the linear warmup is
    learning_rate: float  # maximum learning rate after linear warmup
    min_lr: float  # minimum learning rate after cosine decay
    weight_decay: float  # weight decay coeff for AdamW
    grad_clip: float  # gradient clipping threshold
    eval_interval: int  # number of steps to run before evaluating the model on the validation set
    eval_steps: int  # number of batches to evaluate during validation
    checkpoint_interval: int  # number of steps to run before saving a checkpoint
    checkpoint_dir: str  # directory to save checkpoints to

    @classmethod
    def from_yaml(cls, path: Path) -> TrainingConfig:
        """Load training config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw["training"])
