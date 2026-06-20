"""Tests for ModelConfig architecture fields."""

import pytest

from bob.config import ModelConfig


def make_config(**overrides: object) -> ModelConfig:
    base: dict[str, object] = dict(
        vocab_size=91,
        d_model=48,
        n_heads=3,
        n_layers=3,
        d_ff=192,
        max_seq_len=256,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


def test_d_head_falls_back_to_d_model_over_n_heads() -> None:
    config = make_config(d_model=48, n_heads=3, head_dim=None)
    assert config.d_head == 16


def test_explicit_head_dim_overrides_fallback() -> None:
    # Qwen3 decouples head_dim from d_model // n_heads
    config = make_config(d_model=48, n_heads=4, head_dim=16)
    assert config.d_head == 16  # not 48 // 4 == 12


def test_explicit_head_dim_allows_indivisible_d_model() -> None:
    # with explicit head_dim, d_model need not be divisible by n_heads
    config = make_config(d_model=48, n_heads=5, head_dim=16)
    assert config.d_head == 16


def test_kv_heads_defaults_to_n_heads() -> None:
    config = make_config(n_heads=8, d_model=64, n_kv_heads=None)
    assert config.kv_heads == 8


def test_kv_heads_explicit() -> None:
    config = make_config(n_heads=8, d_model=64, n_kv_heads=2)
    assert config.kv_heads == 2


def test_qk_norm_defaults_false() -> None:
    assert make_config().qk_norm is False


def test_n_heads_not_divisible_by_n_kv_heads_raises() -> None:
    with pytest.raises(ValueError):
        make_config(n_heads=8, d_model=64, n_kv_heads=3)


def test_d_model_not_divisible_by_n_heads_raises_without_head_dim() -> None:
    with pytest.raises(ValueError):
        make_config(d_model=48, n_heads=5, head_dim=None)


def test_odd_d_head_raises() -> None:
    with pytest.raises(ValueError):
        make_config(d_model=48, n_heads=3, head_dim=15)
