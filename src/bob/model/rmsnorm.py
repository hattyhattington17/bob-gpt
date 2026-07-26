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
        # cast to float for numerical stability during sqrt
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.epsilon)  # (B, T, 1)
        dtype = x.dtype  # save original dtype to cast back later
        x = x / rms  # (B, T, d_model)
        # cast back to original dtype and scale by the learned weight parameter gamma
        x = x.to(dtype) * self.gamma  # (B, T, d_model)
        return x
