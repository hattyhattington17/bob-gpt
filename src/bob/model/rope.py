"""Add positional information to query and key vectors by rotating pairs of coordinates."""

import torch


class RoPE(torch.nn.Module):
    """Rotary positional embeddings.

    Precomputes cos/sin buffers for each position j and coordinate pair index n.
    """

    def __init__(self, d_head: int, max_seq_len: int, frequency_base: float) -> None:
        super().__init__()

        # compute the rotation angle theta for each valid sequence position j
        # and coordinate pair index n in {0, ..., d_head//2 - 1}
        # compute sin and cosine of each rotation angle and cache for reuse across heads and batches

        # compute frequency omega_n for each pair of coordinates n in {0, ..., (d_head/2) - 1}
        omegas = frequency_base ** (-torch.arange(0, d_head, 2) / d_head)  # (d_head // 2,)

        positions = torch.arange(max_seq_len)  # (max_seq_len,)

        # θ_{j,n} = j * ω_n for all positions j and pair indices n
        # outer product: (max_seq_len,) x (d_head // 2,)
        thetas = torch.outer(positions, omegas)  # (max_seq_len, d_head // 2)

        # precompute and store cos/sin for each theta
        # persistent=False: not learned params, recomputable from config, excluded from state_dict
        self.rope_cos: torch.Tensor
        self.register_buffer(
            "rope_cos", thetas.cos(), persistent=False
        )  # (max_seq_len, d_head // 2)
        self.rope_sin: torch.Tensor
        self.register_buffer(
            "rope_sin", thetas.sin(), persistent=False
        )  # (max_seq_len, d_head // 2)

    def forward(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return slice of cos and sin cache buffers.

        Starts at offset position, sliced to the current sequence length.

        Args:
            seq_len: Current sequence length, must be <= max_seq_len.
            offset: Position offset for start of current sequence in cos/sin cache

        Returns:
            cos, sin — both shape (seq_len, d_head // 2)
        """
        return self.rope_cos[offset : offset + seq_len], self.rope_sin[offset : offset + seq_len]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a query or key tensor.

    Args:
        x: Query or key tensor, shape (B, n_heads, T, d_head).
        cos: Cached cosine values, shape (T, d_head // 2).
        sin: Cached sine values, shape (T, d_head // 2).

    Returns:
        Tensor with rotary embeddings applied, shape (B, n_heads, T, d_head).
    """
    d_head = x.shape[-1]

    # split input into first half and second half, keep all preceding dimensions (B, n_heads, T)

    # x1 = first half indices (0,...,(d_head // 2)-1)
    x1 = x[..., : d_head // 2]  # (B, n_heads, T, d_head // 2)
    # x2 = second half indices (d_head // 2, ..., d_head - 1)
    x2 = x[..., d_head // 2 :]  # (B, n_heads, T, d_head // 2)

    # d_head // 2 pairs - pair n is x1[n], x2[n] for n in {0, ..., d_head/2 - 1}
    # rotate each pair by theta_{j,n}
    # sin and cos are indexed by position j and pair n — broadcast multiply directly
    # reshape cos/sin to broadcast over batch and head dims
    cos = cos.to(x.dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_head // 2)
    sin = sin.to(x.dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_head // 2)

    # apply rotation: (a cosθ - b sinθ, a sinθ + b cosθ)
    r1 = x1 * cos - x2 * sin  # (B, n_heads, T, d_head // 2)
    r2 = x1 * sin + x2 * cos  # (B, n_heads, T, d_head // 2)

    # concatenate the rotated halves along the last dimension to restore original shape
    return torch.concat([r1, r2], dim=-1)  # (B, n_heads, T, d_head)
