"""Tests for GQA, QK-norm, SDPA, and padding-mask attention."""

import torch

from bob.config import ModelConfig
from bob.inference.kv_cache import KVCache
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


def rope_for(
    config: ModelConfig, absolute_positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rope = RoPE(config.d_head, config.max_seq_len, config.rope_theta)
    # absolute_positions shape (B, T)
    cos, sin = rope(absolute_positions)  # (B, T, d_head // 2)
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
    rope_absolute_positions = torch.arange(seq_len).unsqueeze(0).repeat(2, 1)
    cos, sin = rope_for(config, rope_absolute_positions)
    x = torch.randn(2, seq_len, config.d_model)
    out = attn(
        x=x,
        cos=cos,
        sin=sin,
        layer_index=0,
        padding_mask=torch.ones(2, seq_len, dtype=torch.long),
    )
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
    rope_absolute_positions = torch.arange(seq_len).unsqueeze(0).repeat(2, 1)
    cos, sin = rope_for(make_config(d_model=32, n_heads=8), rope_absolute_positions)
    padding_mask = torch.ones(2, seq_len, dtype=torch.long)

    # different module instances, but both start from gamma=1; copy projection weights
    on.load_state_dict(off.state_dict(), strict=False)
    out_off = off(
        x=x,
        cos=cos,
        sin=sin,
        layer_index=0,
        padding_mask=padding_mask,
    )
    out_on = on(
        x=x,
        cos=cos,
        sin=sin,
        layer_index=0,
        padding_mask=padding_mask,
    )
    assert not torch.allclose(out_off, out_on)


def test_attention_is_causal() -> None:
    config = make_config(n_heads=8, n_kv_heads=2, d_model=32)
    attn = SelfAttention(config)
    seq_len = 6
    rope_absolute_positions = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rope_for(config, rope_absolute_positions)
    x = torch.randn(1, seq_len, config.d_model)
    x2 = x.clone()
    x2[0, -1, :] = torch.randn(config.d_model)  # change only the last position
    padding_mask = torch.ones(1, seq_len, dtype=torch.long)  # no padding
    out = attn(
        x=x,
        cos=cos,
        sin=sin,
        layer_index=0,
        padding_mask=padding_mask,
    )
    out2 = attn(
        x=x2,
        cos=cos,
        sin=sin,
        layer_index=0,
        padding_mask=padding_mask,
    )
    # changing the last token must not affect any earlier query's output
    torch.testing.assert_close(out[:, :-1], out2[:, :-1])


def test_padding_mask_excludes_padded_key() -> None:
    config = make_config(n_heads=8, n_kv_heads=2, d_model=32)
    attn = SelfAttention(config)
    seq_len = 5
    rope_absolute_positions = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rope_for(config, rope_absolute_positions)
    padding_mask = torch.ones(1, seq_len, dtype=torch.long)
    padding_mask[0, 0] = 0  # position 0 is padding

    x = torch.randn(1, seq_len, config.d_model)
    out = attn(
        x=x,
        cos=cos,
        sin=sin,
        layer_index=0,
        padding_mask=padding_mask,
    )

    x2 = x.clone()
    x2[0, 0] = torch.randn(config.d_model)  # garbage in the padded slot
    out2 = attn(
        x=x2,
        cos=cos,
        sin=sin,
        layer_index=0,
        padding_mask=padding_mask,
    )
    # unaffected by garbage in padded slot
    torch.testing.assert_close(out[:, 1:], out2[:, 1:])


def test_padding_branch_stays_causal() -> None:
    # supplying a padding mask must not disable the causal constraint
    config = make_config(n_heads=8, n_kv_heads=2, d_model=32)
    attn = SelfAttention(config)
    seq_len = 6
    rope_absolute_positions = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rope_for(config, rope_absolute_positions)
    padding_mask = torch.ones(1, seq_len, dtype=torch.long)
    padding_mask[0, 0] = 0
    x = torch.randn(1, seq_len, config.d_model)
    x2 = x.clone()
    x2[0, -1, :] = torch.randn(config.d_model)  # change only the last position
    out = attn(
        x=x,
        cos=cos,
        sin=sin,
        layer_index=0,
        padding_mask=padding_mask,
    )
    out2 = attn(
        x=x2,
        cos=cos,
        sin=sin,
        layer_index=0,
        padding_mask=padding_mask,
    )
    # the last (future) token must not affect any earlier query's output
    torch.testing.assert_close(out[:, :-1], out2[:, :-1])


def test_padding_mask_is_per_batch() -> None:
    # mask applies per batch row: padding in row 0 must not touch rows 1, 2
    config = make_config(n_heads=8, n_kv_heads=2, d_model=32)
    attn = SelfAttention(config)
    seq_len = 5
    rope_absolute_positions = torch.arange(seq_len).unsqueeze(0).repeat(3, 1)
    cos, sin = rope_for(config, rope_absolute_positions)
    padding_mask = torch.ones(3, seq_len, dtype=torch.long)
    padding_mask[0, 0] = 0  # only row 0 has a padded position

    x = torch.randn(3, seq_len, config.d_model)
    out = attn(
        x=x,
        cos=cos,
        sin=sin,
        layer_index=0,
        padding_mask=padding_mask,
    )
    assert out.shape == (3, seq_len, config.d_model)

    x2 = x.clone()
    x2[0, 0, :] += 1000.0  # garbage in row 0's padded slot
    out2 = attn(
        x=x2,
        cos=cos,
        sin=sin,
        layer_index=0,
        padding_mask=padding_mask,
    )
    # garbage in the padding slot in batch row 0 doesn't affect the output
    torch.testing.assert_close(out, out2)

    # garbage in a non padded slot in batch row 1 affects output
    x3 = x.clone()
    x3[1, 0, :] += 1000.0  # garbage in row 1
    out3 = attn(
        x=x3,
        cos=cos,
        sin=sin,
        layer_index=0,
        padding_mask=padding_mask,
    )
    assert not torch.allclose(out, out3)


def test_cached_decode_matches_full_forward() -> None:
    # feeding a prompt then one token via cache equals a single full forward
    config = make_config(n_heads=8, n_kv_heads=2, d_model=32, max_seq_len=16)
    attn = SelfAttention(config).eval()
    prefill_length = 4
    full_sequence_length = prefill_length + 1

    rope_absolute_positions = torch.arange(full_sequence_length).unsqueeze(0)
    padding_mask = torch.ones(1, full_sequence_length, dtype=torch.long)

    x_full = torch.randn(1, full_sequence_length, config.d_model)
    cos, sin = rope_for(config, rope_absolute_positions)
    out_full = attn(
        x=x_full,
        cos=cos,
        sin=sin,
        layer_index=0,
        padding_mask=padding_mask,
    )

    # run prefill against the cache
    cache = KVCache(config, batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    out1 = attn(
        x=x_full[:, :prefill_length],
        cos=cos[:, :prefill_length],
        sin=sin[:, :prefill_length],
        layer_index=0,
        padding_mask=padding_mask[:, :prefill_length],
        past_seen_tokens=0,
        kv_cache=cache,
    )
    assert cache.kv_length == prefill_length
    torch.testing.assert_close(out1, out_full[:, :prefill_length])

    out_step = attn(
        x=x_full[:, -1:],
        cos=cos[:, -1:],
        sin=sin[:, -1:],
        layer_index=0,
        padding_mask=padding_mask,
        past_seen_tokens=prefill_length,
        kv_cache=cache,
    )

    # the cached step's single output must match the last position of the full forward
    torch.testing.assert_close(out_step[:, 0], out_full[:, -1])
