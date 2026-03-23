"""Multilayer perceptron/feed-forward network."""

import torch

from bob.config import ModelConfig


class MLP(torch.nn.Module):
    """Multilayer perceptron/feed-forward network.

    Transforms an individual positions hidden state independently via a gated feed forward network.
    Uses SiLU nonlinearity and a gated linear unit (GLU).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # W_a, W_b shape (d_model, d_ff)
        self.W_a = torch.nn.Linear(config.d_model, config.d_ff, bias=False)
        self.W_b = torch.nn.Linear(config.d_model, config.d_ff, bias=False)
        # W_out shape (d_ff, d_model)
        self.W_out = torch.nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute MLP output per position.

        Args:
            x: Normalized hidden state, shape (B, T, d_model).

        Returns:
            MLP output, shape (B, T, d_model).
        """
        # project into gate and candidate feature tensors a,b shape (B,T, d_ff)
        a = self.W_a(x)
        b = self.W_b(x)

        # construct gate vector by applying SiLU nonlinearity to b
        b = b * torch.sigmoid(b)  # (B, T, d_ff)

        # apply to candidate features via elementwise multiplication
        g = a * b  # (B, T, d_ff)

        # project back to model dimension
        g = self.W_out(g)  # (B, T, d_model)

        return g
