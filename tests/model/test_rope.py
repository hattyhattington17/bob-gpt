"""Tests for rotate_half RoPE."""

import math

import torch

from bob.model.rope import RoPE, apply_rotary_emb


def test_rope_position_zero_is_identity() -> None:
    rope = RoPE(d_head=8, max_seq_len=16, frequency_base=10000.0)
    cos, sin = rope(4)
    x = torch.randn(2, 3, 4, 8)  # (B, n_heads, T, d_head)
    out = apply_rotary_emb(x, cos, sin)
    # theta at position 0 is 0, so the first sequence position is unrotated
    torch.testing.assert_close(out[:, :, 0, :], x[:, :, 0, :])


def test_rope_preserves_vector_norm() -> None:
    rope = RoPE(d_head=8, max_seq_len=16, frequency_base=10000.0)
    cos, sin = rope(5)
    x = torch.randn(2, 3, 5, 8)
    out = apply_rotary_emb(x, cos, sin)
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1))


def test_rope_uses_rotate_half_pairing() -> None:
    # Pin the convention: coordinate m pairs with m + d_head // 2 (rotate_half)
    base = 10000.0
    d_head = 4
    half = d_head // 2
    seq_len = 3
    rope = RoPE(d_head=d_head, max_seq_len=8, frequency_base=base)
    cos, sin = rope(seq_len)

    x = torch.arange(1, seq_len * d_head + 1, dtype=torch.float32).reshape(1, 1, seq_len, d_head)
    out = apply_rotary_emb(x, cos, sin)

    omegas = [base ** (-(2 * m) / d_head) for m in range(half)]
    expected = torch.zeros_like(x)
    for j in range(seq_len):
        for m in range(half):
            theta = j * omegas[m]
            a = x[0, 0, j, m].item()  # first-half coordinate
            b = x[0, 0, j, m + half].item()  # paired second-half coordinate
            expected[0, 0, j, m] = a * math.cos(theta) - b * math.sin(theta)
            expected[0, 0, j, m + half] = a * math.sin(theta) + b * math.cos(theta)

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_rope_relative_position_invariance() -> None:
    # RoPE's defining property: <rotate(u, m), rotate(u, n)> depends only on m - n.
    rope = RoPE(d_head=8, max_seq_len=16, frequency_base=10000.0)
    cos, sin = rope(10)
    u = torch.randn(1, 1, 1, 8)
    x = u.expand(1, 1, 10, 8).contiguous()  # same vector at every position
    rotated = apply_rotary_emb(x, cos, sin)  # (1, 1, 10, 8)

    def score(m: int, n: int) -> float:
        return float((rotated[0, 0, m] * rotated[0, 0, n]).sum())

    torch.testing.assert_close(
        torch.tensor(score(5, 2)), torch.tensor(score(8, 5)), rtol=1e-4, atol=1e-4
    )
