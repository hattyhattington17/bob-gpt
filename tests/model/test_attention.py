"""Tests for GQA, QK-norm, SDPA, and padding-mask attention."""

import torch

from bob.config import ModelConfig
from bob.model.attention import SelfAttention, repeat_kv
from bob.model.rope import RoPE


def make_config(**overrides: object) -> ModelConfig:
    base: dict[str, object] = dict(
        vocab_size=50,
        d_model=32,
        n_heads=8,
        n_layers=1,
        d_ff=64,
        max_seq_len=16,
    )
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


def rope_for(config: ModelConfig, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    rope = RoPE(config.d_head, config.max_seq_len, config.rope_theta)
    cos, sin = rope(seq_len)
    return cos, sin


def test_repeat_kv_expands_each_head_consecutively() -> None:
    x = torch.randn(2, 2, 5, 4)  # (B, n_kv_heads, T, d_head)
    out = repeat_kv(x, G=3)
    assert out.shape == (2, 6, 5, 4)
    for h in range(6):
        torch.testing.assert_close(out[:, h], x[:, h // 3])


def test_repeat_kv_identity_when_n_rep_one() -> None:
    x = torch.randn(2, 4, 5, 4)
    torch.testing.assert_close(repeat_kv(x, G=1), x)


def test_kv_projections_sized_by_kv_heads() -> None:
    config = make_config(n_heads=8, n_kv_heads=2, d_model=32)
    attn = SelfAttention(config)
    # q spans all heads, k/v span only kv heads
    assert attn.W_q.weight.shape == (8 * config.d_head, 32)
    assert attn.W_k.weight.shape == (2 * config.d_head, 32)
    assert attn.W_v.weight.shape == (2 * config.d_head, 32)


def test_gqa_forward_shape() -> None:
    config = make_config(n_heads=8, n_kv_heads=2, d_model=32)
    attn = SelfAttention(config)
    seq_len = 6
    cos, sin = rope_for(config, seq_len)
    x = torch.randn(2, seq_len, config.d_model)
    out = attn(x, cos, sin)
    assert out.shape == (2, seq_len, config.d_model)
