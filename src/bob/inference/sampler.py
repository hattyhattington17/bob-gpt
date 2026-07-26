"""Sampling strategies for token generation."""

import torch


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """Select the highest probability token.

    Args:
        logits: Logits for each sequence position, shape (B, T, vocab_size).

    Returns:
        Token id of the highest probability token for each batch element.
    """
    next_token_logits = logits[:, -1, :]  # (B, vocab_size)
    # return the index of the largest logit
    return next_token_logits.argmax(-1, keepdim=True)  # (B, 1)
