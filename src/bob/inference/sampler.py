import torch


def greedy(logits: torch.Tensor) -> int:
    """Select the highest probability token.

    Args:
        logits: shape (vocab_size,) — the logits for the last position in the sequence, they predict the next token to generate
    Returns:
        token id of the highest probability token
    """
    # return the index of the largest logit
    return int(logits.argmax().item())
