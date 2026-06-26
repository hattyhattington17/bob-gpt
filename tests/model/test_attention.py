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


def test_qk_norm_modules_none_when_disabled() -> None:
    attn = SelfAttention(make_config(qk_norm=False))
    assert attn.q_norm is None
    assert attn.k_norm is None


def test_qk_norm_modules_created_when_enabled() -> None:
    config = make_config(qk_norm=True)
    attn = SelfAttention(config)
    assert attn.q_norm is not None
    assert attn.k_norm is not None
    # head-wise: normalizes over d_head, not d_model
    assert attn.q_norm.gamma.shape == (config.d_head,)


def test_qk_norm_gives_unit_rms_per_head() -> None:
    config = make_config(qk_norm=True)
    attn = SelfAttention(config)
    assert attn.q_norm is not None
    q = torch.randn(2, config.n_heads, 5, config.d_head) * 7.0
    normed = attn.q_norm(q)
    rms = normed.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-3, atol=1e-3)


def test_qk_norm_changes_forward_output() -> None:
    seq_len = 6
    x = torch.randn(2, seq_len, 32)
    off = SelfAttention(make_config(qk_norm=False, d_model=32, n_heads=8))
    on = SelfAttention(make_config(qk_norm=True, d_model=32, n_heads=8))
    cos, sin = rope_for(make_config(d_model=32, n_heads=8), seq_len)
    # different module instances, but both start from gamma=1; copy projection weights
    on.load_state_dict(off.state_dict(), strict=False)
    out_off = off(x, cos, sin)
    out_on = on(x, cos, sin)
    assert not torch.allclose(out_off, out_on)


def test_attention_is_causal() -> None:
    config = make_config(n_heads=8, n_kv_heads=2, d_model=32)
    attn = SelfAttention(config)
    seq_len = 6
    cos, sin = rope_for(config, seq_len)
    x = torch.randn(1, seq_len, config.d_model)
    x2 = x.clone()
    x2[0, -1, :] = torch.randn(config.d_model)  # change only the last position
    out = attn(x, cos, sin)
    out2 = attn(x2, cos, sin)
    # changing the last token must not affect any earlier query's output
    torch.testing.assert_close(out[:, :-1], out2[:, :-1])


def test_padding_mask_excludes_padded_key() -> None:
    config = make_config(n_heads=8, n_kv_heads=2, d_model=32)
    attn = SelfAttention(config)
    seq_len = 5
    cos, sin = rope_for(config, seq_len)
    padding_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    padding_mask[0, 1] = True  # position 1 is padding

    x = torch.randn(1, seq_len, config.d_model)
    out = attn(x, cos, sin, padding_mask=padding_mask)
    x2 = x.clone()
    x2[0, 1, :] += 1000.0  # garbage in the padded slot
    out2 = attn(x2, cos, sin, padding_mask=padding_mask)
    # queries 2..4 attend over key 1 only if unmasked; with it masked they are unaffected
    torch.testing.assert_close(out[:, 2:], out2[:, 2:])


def test_no_padding_mask_matches_default() -> None:
    config = make_config(n_heads=8, n_kv_heads=2, d_model=32)
    attn = SelfAttention(config)
    seq_len = 5
    cos, sin = rope_for(config, seq_len)
    x = torch.randn(1, seq_len, config.d_model)
    out_default = attn(x, cos, sin)
    out_none = attn(x, cos, sin, padding_mask=None)
    torch.testing.assert_close(out_default, out_none)


def test_padding_branch_stays_causal() -> None:
    # supplying a padding mask must not disable the causal constraint
    config = make_config(n_heads=8, n_kv_heads=2, d_model=32)
    attn = SelfAttention(config)
    seq_len = 6
    cos, sin = rope_for(config, seq_len)
    padding_mask = torch.zeros(1, seq_len, dtype=torch.bool)  # no padding
    x = torch.randn(1, seq_len, config.d_model)
    x2 = x.clone()
    x2[0, -1, :] = torch.randn(config.d_model)  # change only the last position
    out = attn(x, cos, sin, padding_mask=padding_mask)
    out2 = attn(x2, cos, sin, padding_mask=padding_mask)
    # the last (future) token must not affect any earlier query's output
    torch.testing.assert_close(out[:, :-1], out2[:, :-1])


def test_padding_mask_is_per_batch() -> None:
    # mask applies per batch row: padding in row 0 must not touch rows 1, 2
    config = make_config(n_heads=8, n_kv_heads=2, d_model=32)
    attn = SelfAttention(config)
    seq_len = 5
    cos, sin = rope_for(config, seq_len)
    padding_mask = torch.zeros(3, seq_len, dtype=torch.bool)
    padding_mask[0, 1] = True  # only row 0 has a padded position

    x = torch.randn(3, seq_len, config.d_model)
    out = attn(x, cos, sin, padding_mask=padding_mask)
    assert out.shape == (3, seq_len, config.d_model)

    x2 = x.clone()
    x2[0, 1, :] += 1000.0  # garbage in row 0's padded slot
    out2 = attn(x2, cos, sin, padding_mask=padding_mask)
    # rows 1 and 2 share no keys with row 0, so they must be unchanged
    torch.testing.assert_close(out[1:], out2[1:])
