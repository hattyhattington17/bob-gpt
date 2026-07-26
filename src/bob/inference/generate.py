"""Autoregressive token generation."""

import torch

from bob.inference.kv_cache import KVCache
from bob.inference.sampler import greedy
from bob.model.transformer import Bob


# disable gradient computation
@torch.inference_mode()
def generate(
    model: Bob,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Batched autoregressive token generation with incremental decode using a KV cache.

    Args:
        model: The model in eval mode.
        input_ids: Left padded tensor of prefill sequence token ids (B,T).
        max_new_tokens: Max number of tokens to generate.

    Returns:
        Full generated sequence including prompt as a tensor of token ids (B, T + max_new_tokens).
    """
    if input_ids.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError(
            f"Prompt length {input_ids.shape[1]} plus max_new_tokens {max_new_tokens} "
            f"exceeds max_seq_len {model.config.max_seq_len}. "
            "Truncate the prompt or increase max_seq_len."
        )

    param = next(model.parameters())
    # initialize KV cache for incremental decode
    kv_cache = KVCache(
        model.config, batch_size=input_ids.shape[0], device=param.device, dtype=param.dtype
    )

    # todo - padding mask should be supplied by tokenizer
    # append for each new token generated, pass through to model to compute absolute positions
    padding_mask = torch.ones_like(input_ids, dtype=torch.long)  # (B, T)

    # prefill - processes whole prefill sequence and fills KV cache for incremental decode
    model_input = input_ids  # (B, T)
    # autoregressive decode loop
    for _ in range(max_new_tokens):
        logits = model(
            model_input, padding_mask=padding_mask, kv_cache=kv_cache
        )  # (B, T, vocab_size), first prefill round will have T=prefill length, after that T=1

        # todo - implement other sampling strategies and EOS token handling
        model_input = greedy(logits)  # (B, 1)

        # append the sampled token to the sequence
        input_ids = torch.cat((input_ids, model_input), dim=1)  # (B, kv_length + 1)
        padding_mask = torch.cat(
            (padding_mask, torch.ones_like(model_input, dtype=torch.long)), dim=1
        )

    return input_ids  # (B, T + max_new_tokens)
