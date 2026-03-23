"""Sampling strategies for token generation."""

import torch


def greedy(logits: torch.Tensor) -> int:
    """Select the highest probability token.

    Args:
        logits: Logits for the next token position, shape (vocab_size,).

    Returns:
        Token id of the highest probability token.
    """
    # return the index of the largest logit
    return int(logits.argmax().item())
