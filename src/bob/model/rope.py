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
        omegas = frequency_base ** (-torch.arange(0, d_head, 2).float()/d_head)


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
    return x
