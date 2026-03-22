import torch


class RMSNorm(torch.nn.Module):
    """RMSNorm - Root Mean Square Layer Normalization
    Normalizes the input tensor across the last dimension to have a fixed RMS value
    then scales by a learned weight parameter
    """

    def __init__(self, d_model: int, epsilon: float) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(d_model))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input tensor x in R^{B, T, d_model} across the last dimension d_model
        to have a fixed RMS value, then scale by the learned weight parameter gamma."""

        # compute the RMS for each position in the sequence and batch
        # RMS is the square root of the mean of the squares of the values across the last dimension
        # rms in R^{B, T, 1} - keepdim=True to maintain the same number of dimensions for broadcasting
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        x = x / rms
        # scale by the learned weight parameter
        x = x * self.gamma
        return x
