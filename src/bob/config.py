"""Load and validate model configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import yaml


@dataclass(frozen=True, slots=True)  # immutable
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
    qk_norm: bool = False
    head_dim: int | None = None
    n_kv_heads: int | None = None

    @property
    def d_head(self) -> int:
        """Return the per-head dimension.

        If head_dim was not provided in the config, this falls back to d_model // n_heads.
        """
        return self.head_dim if self.head_dim is not None else self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        """Return the number of key-value heads.

        If n_kv_heads was not provided in the config, this falls back to n_heads.
        """
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    def __post_init__(self) -> None:
        if self.head_dim is None and self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}) "
                "when head_dim is not set"
            )

        if self.d_head % 2 != 0:
            raise ValueError(f"d_head ({self.d_head}) must be even for RoPE")

        if self.n_kv_heads is not None and self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        """Load model config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw["model"])


@dataclass(frozen=True, slots=True)
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
    checkpoint_dir: str  # directory to save checkpoints to

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        """Load training config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw["training"])
