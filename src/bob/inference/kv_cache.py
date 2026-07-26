"""KV cache."""

import torch

from bob.config import ModelConfig


class KVCache:
    """Key-Value cache for incremental decode in transformer self attention.

    Stores KV projections for each layer for all cached positions up to max_seq_len.

    k_cache[layer], v_cache[layer] shape: (B, n_kv_heads, max_seq_len, d_head)
        preallocated to max_seq_len but only kv_length positions are valid for attention.
    Keys have already been processed with QKNorm and RoPE.
    All sequences in a batch share a single kv_length. Callers must left pad
        batched prompts to equal length before prefill.
    Supplies KV projections for previous positions to attention sublayers.

    During a forward pass, kv_length grows from past_seen_tokens -> past_seen_tokens + T,
        where T is the model input sequence length.
    """

    def __init__(
        self,
        config: ModelConfig,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the KV cache.

        Args:
            config: Model configuration.
            batch_size: Batch size.
            device: Device to place cache tensors on.
            dtype: Data type for cache tensors. Defaults to torch.float32.
        """
        self.n_layers = config.n_layers
        self.max_seq_len = config.max_seq_len
        self.kv_length = 0

        # preallocate buffer to max_seq_len to avoid reallocating during incremental decode
        self.k_cache: list[torch.Tensor] = [
            torch.zeros(
                batch_size,
                config.kv_heads,
                config.max_seq_len,
                config.d_head,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.n_layers)
        ]  # (B, n_kv_heads, max_seq_len, d_head)
        self.v_cache: list[torch.Tensor] = [
            torch.zeros(
                batch_size,
                config.kv_heads,
                config.max_seq_len,
                config.d_head,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.n_layers)
        ]  # (B, n_kv_heads, max_seq_len, d_head)

    # todo - consider per-layer internal length tracking (HF StaticLayer.cumulative_length) so
    # past_seen_tokens no longer threads through Bob -> TransformerBlock -> SelfAttention -> update
    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, past_seen_tokens: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the KV cache for a given layer with new key and value projections.

        Args:
            layer: Index of the transformer layer to update.
            k: Key projections, shape (B, n_kv_heads, update_length, d_head).
            v: Value projections, shape (B, n_kv_heads, update_length, d_head).
            past_seen_tokens: Position in the cache to start writing new keys and values.

        Returns:
            Tuple of updated key and value caches for the specified layer, each with shape
            (B, n_kv_heads, kv_length, d_head).
        """
        if layer < 0 or layer >= self.n_layers:
            raise ValueError(f"Layer index {layer} is out of bounds for n_layers={self.n_layers}")

        if past_seen_tokens < 0 or past_seen_tokens + k.shape[2] > self.max_seq_len:
            raise ValueError(
                f"Past seen tokens {past_seen_tokens} with update length {k.shape[2]} "
                f"exceeds max_seq_len={self.max_seq_len}"
            )

        # update new keys and values to the existing cache tensor for the specified layer
        self.k_cache[layer][:, :, past_seen_tokens : past_seen_tokens + k.shape[2], :] = k
        self.v_cache[layer][:, :, past_seen_tokens : past_seen_tokens + v.shape[2], :] = v
        if layer == 0:
            # update kv_length only once per forward pass
            self.kv_length = past_seen_tokens + k.shape[2]

        # (B, n_kv_heads, kv_length, d_head)
        return (
            self.k_cache[layer][:, :, : self.kv_length, :],
            self.v_cache[layer][:, :, : self.kv_length, :],
        )
