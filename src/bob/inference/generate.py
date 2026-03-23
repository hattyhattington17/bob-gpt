"""Autoregressive token generation."""

import torch

from bob.inference.sampler import greedy
from bob.model.transformer import Bob


# disable gradient tracking since we're only doing inference    
@torch.inference_mode()
def generate(
    model: Bob,
    token_ids: list[int],
    max_new_tokens: int,
    max_seq_len: int,
    device: str,
) -> list[int]:
    """Autoregressively generate tokens given a prompt.

    Args:
        model: The Bob model in eval mode (training behavior like dropout is disabled).
        token_ids: List of token ids with shape (T,).
        max_new_tokens: Max number of tokens to generate.
        max_seq_len: Maximum context window length, truncates input if exceeded.
        device: Device to run the model on, e.g. "cpu" or "mps".

    Returns:
        Full sequence including prompt as a list of token ids.
    """
    # batch dimension of 1 added since the model expects batches
    x = torch.tensor([token_ids], dtype=torch.long).to(device)  # (1, T)

    for _ in range(max_new_tokens):
        # take last max_seq_len tokens as input to the model
        x = x[:, -max_seq_len:]  # (1, T)
        logits = model(x)  # (1, T, vocab_size)
        # slice to the last position and sample the next token id
        next_token = greedy(logits[0, -1, :])  # logits[0, -1, :] is (vocab_size,)
        x = torch.cat([x, torch.tensor([[next_token]], device=x.device)], dim=1)  # (1, T+1)

    return x[0].tolist()
