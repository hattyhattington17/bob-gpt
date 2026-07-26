"""Add positional information to query and key vectors by rotating pairs of coordinates."""

import torch


class RoPE(torch.nn.Module):
    """Rotary positional embeddings.

    Precomputes cos/sin buffers for each angle theta_{j,n} = j * omega_n for each
        absolute position j in {0, ..., max_seq_len - 1} and
        coordinate pair index n in {0, ..., (d_head//2) - 1}.
    """

    def __init__(self, d_head: int, max_seq_len: int, frequency_base: float) -> None:
        super().__init__()

        # compute rotation frequency omega_n = frequency_base^(-2n/d_head)
        #  for each pair index n in {0, ..., (d_head//2) - 1}
        omegas = frequency_base ** (-torch.arange(0, d_head, 2) / d_head)  # (d_head // 2)

        # compute rotation angles (radians) theta_{j,n} = j * omega_n
        #   for each valid absolute position j in {0, ..., max_seq_len - 1}
        positions = torch.arange(max_seq_len)  # (max_seq_len)
        thetas = torch.outer(positions, omegas)  # (max_seq_len, d_head // 2)

        # precompute and store cos/sin for each theta
        self.rope_cos: torch.Tensor
        # store tensor as module state but not a parameter (optimizer does not update)
        # persistent=False excluded from checkpoint state_dict
        self.register_buffer(
            "rope_cos", thetas.cos(), persistent=False
        )  # (max_seq_len, d_head // 2)
        self.rope_sin: torch.Tensor
        self.register_buffer(
            "rope_sin", thetas.sin(), persistent=False
        )  # (max_seq_len, d_head // 2)

    def forward(self, rope_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return entries in cos and sin cache buffers for the absolute positions being processed.

        Args:
            rope_positions: Absolute positions for the current batch of sequences, shape (B, T).

        Returns:
            cos and sin buffers for the current batch of sequences, each shaped (B, T, d_head // 2).
        """
        return self.rope_cos[rope_positions], self.rope_sin[rope_positions]  # (B, T, d_head // 2)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a query or key tensor.

    Splits the last dimension of the tensor in half, combines into pairs, and rotates each pair by
    the angle theta_{j,n} for the corresponding absolute position j and pair index n.

    Args:
        x: Query or key tensor, shape (B, n_heads, T, d_head).
        cos: Cached cosine values, shape (B, T, d_head // 2).
        sin: Cached sine values, shape (B, T, d_head // 2).

    Returns:
        Tensor with rotary embeddings applied, shape (B, n_heads, T, d_head).
    """
    d_head = x.shape[-1]

    # split last dimension into first half x1 and second half x2, construct d_head // 2 pairs
    # position j pair n is (x1[..., j, n], x2[..., j, n]) for n in {0, ..., d_head/2 - 1}

    # x1 = first half
    x1 = x[..., : d_head // 2]  # (B, n_heads, T, d_head // 2)
    # x2 = second half
    x2 = x[..., d_head // 2 :]  # (B, n_heads, T, d_head // 2)

    # rotate each pair (x1[..., j, n], x2[..., j, n]) by the angle theta_{j,n} using cached cos/sin
    # broadcast rotation across all heads (absolute positions vary by sequence in batch)
    cos = cos.unsqueeze(1)  # (B, 1, T, d_head // 2)
    sin = sin.unsqueeze(1)  # (B, 1, T, d_head // 2)

    # apply rotation: (a cosθ - b sinθ, a sinθ + b cosθ)
    r1 = x1 * cos - x2 * sin  # (B, n_heads, T, d_head // 2)
    r2 = x1 * sin + x2 * cos  # (B, n_heads, T, d_head // 2)

    # concatenate the rotated halves along the last dimension to restore original shape
    # cos/sin are float32, cast back to original datatype
    return torch.cat([r1, r2], dim=-1).to(x.dtype)  # (B, n_heads, T, d_head)
