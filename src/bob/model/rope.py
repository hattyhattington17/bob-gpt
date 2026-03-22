"""Add positional information to query and key vectors by rotating pairs of coordinates"""

import torch


class RoPE(torch.nn.Module):
    """Rotary positional embeddings
    Precomputes cos/sin buffers for each position j and coordinate pair index m
    """

    def __init__(self, d_head: int, max_seq_len: int, frequency_base: float) -> None:
        super().__init__()

        # compute the rotation angle theta for each valid sequence position j
        # and coordinate pair index m in {0, ..., d_head//2 - 1}
        # compute sin and cosine of each rotation angle and cache for use across attention heads and batches

        # compute frequency omega_m for each pair of coordinates m in {0, ..., (d_head/2) - 1}
        # omegas in R^{d_head/2}
        omegas = frequency_base ** (-torch.arange(0, d_head, 2).float() / d_head)

        # position indices j in R^{max_seq_len}
        positions = torch.arange(max_seq_len).float()

        # θ_{j,m} = j * ω_m for all positions j and pair indices m
        # outer product: j in R^{max_seq_len} x omegas in R^{d_head / 2}
        # thetas in R^{max_seq_len, d_head / 2}
        thetas = torch.outer(positions, omegas)

        # precompute and store cos/sin for each theta
        # persistent=False because these aren't learned params and shouldn't be stored with the model params
        self.register_buffer("rope_cos", thetas.cos(), persistent=False)
        self.register_buffer("rope_sin", thetas.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin cache buffers sliced to the current sequence length.

        Returns:
            cos, sin — both shape (seq_len, d_head // 2)
        """
        return self.rope_cos[:seq_len], self.rope_sin[:seq_len]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """ 
    Apply RoPE to a Query or Key tensor in R^{B, H, C, d_head} 
    using cached cos and sin buffers in R^{T, d_head/2} 
    """
    
    # split input into consecutive coordinate pairs
    # x1 = even indices (0, 2, 4, ...), x2 = odd indices (1, 3, 5, ...)
    # keep all preceding dimensions (B, H, C) 

    # split only the last dimension d_head into pairs
    x1 = x[..., ::2]   # (B, H, C, d_head / 2)
    # offset by 1 to get odd indices
    x2 = x[..., 1::2]  # (B, H, C, d_head / 2)

    # each pair is x1[i], x2[i] for i in {0, ..., d_head/2 - 1}
    # rotate each pair by theta for the position j and pair index i
    # sin and cos buffers are already indexed by position j and pair index i, so we can just broadcast multiply
    
    # reshape cos/sin to broadcast over batch and head dims
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, C, d_head // 2)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, C, d_head // 2)

    # apply rotation: (a cosθ - b sinθ, a sinθ + b cosθ)
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos

    # r1, r2 shape: (B, H, C, d_head / 2)
    # stack along a new last dimension to pair each rotated coordinate back together
    # (B, H, C, d_head / 2, 2) — last dim is [r1_m, r2_m] for each pair m
    stacked = torch.stack([r1, r2], dim=-1)

    # flatten the last two dims to merge pairs back into a flat vector
    # (B, H, C, d_head / 2, 2) → (B, H, C, d_head)
    # result has rotated pairs interleaved: (r1_0, r2_0, r1_1, r2_1, ...)
    return stacked.flatten(-2)
