"""Root Mean Square Layer Normalization."""

import torch


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes the input tensor across the last dimension to have a fixed RMS
    value, then scales by a learned weight parameter.
    """

    def __init__(self, d_model: int, epsilon: float) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(d_model))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x and scale by the learned weight parameter gamma.

        Args:
            x: Input tensor, shape (B, T, d_model).

        Returns:
            Normalized tensor, shape (B, T, d_model).
        """
        # compute RMS across the last dimension
        # keepdim=True to maintain dimensions for broadcasting
        # cast to float for numerical stability during sqrt, then back to original dtype
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.epsilon).to(
            x.dtype
        )  # (B, T, 1)
        x = x / rms  # (B, T, d_model)
        # scale by the learned weight parameter
        x = x * self.gamma  # (B, T, d_model)
        return x
