"""
loads the config and validates it
"""

from __future__ import annotations
from dataclasses import dataclass
import yaml


@dataclass(frozen=True) # immutable
class ModelConfig:
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
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.d_head % 2 == 0, "d_head must be even for RoPE"

    @classmethod
    def from_yaml(cls, path: str) -> ModelConfig:
        """Load model config from a YAML file"""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw["model"])