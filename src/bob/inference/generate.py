"""Autoregressive token generation."""

import torch

from bob.inference.kv_cache import KVCache
from bob.model.transformer import Bob


# disable gradient tracking since we're only doing inference
@torch.inference_mode()
def generate(
    model: Bob,
    token_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Autoregressively generate tokens given a prompt.

    Args:
        model: The Bob model in eval mode (training behavior like dropout is disabled).
        token_ids: Tensor of prefill token ids (B,T).
        max_new_tokens: Max number of tokens to generate.

    Returns:
        Full sequence including prompt as a list of token ids.
    """
    params = next(model.parameters())
    # initialize KV cache for incremental decode
    kv_cache = KVCache(
        model.config, batch_size=token_ids.shape[0], device=params.device, dtype=params.dtype
    )

    if token_ids.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError(
            f"Prompt length {token_ids.shape[1]} exceeds max_seq_len {model.config.max_seq_len}. "
            "Truncate the prompt or increase max_seq_len."
        )
    # todo - tokenizer should left pad prefill sequences to the longest sequence length in the batch
    # generate padding mask
    # append for each new token generated, pass through to model to compute absolute positions
    token_sequence = token_ids  # (B, T)
    padding_mask = torch.ones_like(token_sequence, dtype=params.dtype)  # (B, T)

    # prefill - processes whole prefill sequence and fills KV cache for incremental decode
    next_tokens = token_sequence  # (B, T)
    # autoregressive decode loop
    for _ in range(max_new_tokens):
        logits = model(
            next_tokens, padding_mask=padding_mask, kv_cache=kv_cache
        )  # (B, 1, vocab_size)

        # todo - implement other sampling strategies and EOS token handling
        next_tokens = logits[:, -1, :].argmax(-1, keepdim=True)  # (B, 1)

        # append the sampled token to the sequence
        token_sequence = torch.cat((token_sequence, next_tokens), dim=1)
        padding_mask = torch.cat(
            (padding_mask, torch.ones_like(next_tokens, dtype=params.dtype)), dim=1
        )

    return token_sequence  # (B, T + max_new_tokens)
