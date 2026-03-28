"""Add positional information to query and key vectors by rotating pairs of coordinates."""

import torch


class RoPE(torch.nn.Module):
    """Rotary positional embeddings.

    Precomputes cos/sin buffers for each position j and coordinate pair index m.
    """

    def __init__(self, d_head: int, max_seq_len: int, frequency_base: float) -> None:
        super().__init__()

        # compute the rotation angle theta for each valid sequence position j
        # and coordinate pair index m in {0, ..., d_head//2 - 1}
        # compute sin and cosine of each rotation angle and cache for reuse across heads and batches

        # compute frequency omega_m for each pair of coordinates m in {0, ..., (d_head/2) - 1}
        omegas = frequency_base ** (-torch.arange(0, d_head, 2) / d_head)  # (d_head // 2,)

        positions = torch.arange(max_seq_len)  # (max_seq_len,)

        # θ_{j,m} = j * ω_m for all positions j and pair indices m
        # outer product: (max_seq_len,) x (d_head // 2,)
        thetas = torch.outer(positions, omegas)  # (max_seq_len, d_head // 2)

        # precompute and store cos/sin for each theta
        # persistent=False: not learned params, recomputable from config, excluded from state_dict
        self.register_buffer(
            "rope_cos", thetas.cos(), persistent=False
        )  # (max_seq_len, d_head // 2)
        self.register_buffer(
            "rope_sin", thetas.sin(), persistent=False
        )  # (max_seq_len, d_head // 2)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin cache buffers sliced to the current sequence length.

        Returns:
            cos, sin — both shape (seq_len, d_head // 2)
        """
        return self.rope_cos[:seq_len], self.rope_sin[:seq_len]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a query or key tensor.

    Args:
        x: Query or key tensor, shape (B, n_heads, T, d_head).
        cos: Cached cosine values, shape (T, d_head // 2).
        sin: Cached sine values, shape (T, d_head // 2).

    Returns:
        Tensor with rotary embeddings applied, shape (B, n_heads, T, d_head).
    """
    # split input into consecutive coordinate pairs
    # x1 = even indices (0, 2, 4, ...), x2 = odd indices (1, 3, 5, ...)
    # keep all preceding dimensions (B, n_heads, T)
    x1 = x[..., ::2]  # (B, n_heads, T, d_head // 2)
    x2 = x[..., 1::2]  # (B, n_heads, T, d_head // 2)

    # each pair is x1[i], x2[i] for i in {0, ..., d_head/2 - 1}
    # rotate each pair by theta for the position j and pair index i
    # sin and cos are indexed by position j and pair index i — broadcast multiply directly

    # reshape cos/sin to broadcast over batch and head dims
    cos = cos.to(x.dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_head // 2)
    sin = sin.to(x.dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_head // 2)

    # apply rotation: (a cosθ - b sinθ, a sinθ + b cosθ)
    r1 = x1 * cos - x2 * sin  # (B, n_heads, T, d_head // 2)
    r2 = x1 * sin + x2 * cos  # (B, n_heads, T, d_head // 2)

    # stack along a new last dimension — last dim is [r1_m, r2_m] for each pair m
    stacked = torch.stack([r1, r2], dim=-1)  # (B, n_heads, T, d_head // 2, 2)

    # flatten the last two dims to merge pairs back into a flat vector
    # result has rotated pairs interleaved: (r1_0, r2_0, r1_1, r2_1, ...)
    return stacked.flatten(-2)  # (B, n_heads, T, d_head)
