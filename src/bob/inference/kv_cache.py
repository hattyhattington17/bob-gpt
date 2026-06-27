"""KV cache."""

import torch

from bob.config import ModelConfig


class KVCache:
    """Key-Value cache for autoregressive generation.

    Stores Key and Value projections after applying QKNorm and RoPE for each position at each layer.
    Allows efficient incremental decode by supplying previous KV projections to attention sublayers.
    """

    def __init__(
        self,
        config: ModelConfig,
        batch_size: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the KV cache.

        For each transformer layer, stores a tensor of shape
            (B, n_kv_heads, sequence_length, d_head)
        where sequence_length is the number of token positions cached so far. For batched prefill,
        shorter sequences are padded so sequence_length is uniform across sequences in the batch

        Args:
            config: Model configuration.
            batch_size: Batch size for batched prefill.
            device: Device to place cache tensors on. Defaults to CPU.
            dtype: Data type for cache tensors. Defaults to torch.float32.
        """
        if device is None:
            device = torch.device("cpu")
        self.n_layers = config.n_layers
        self.max_seq_len = config.max_seq_len

        self.sequence_length = 0

        # preallocate the full buffer for each layer to avoid reallocating during incremental decode
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
        ]  # (B, n_kv_heads, sequence_length, d_head)
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
        ]  # (B, n_kv_heads, sequence_length, d_head)

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, cache_position: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the KV cache for a given layer with new key and value projections.

        Args:
            layer: Index of the transformer layer to update.
            k: Key projections, shape (B, n_kv_heads, update_length, d_head).
            v: Value projections, shape (B, n_kv_heads, update_length, d_head).
            cache_position: Position in the cache to start writing new keys and values.
        """
        if layer < 0 or layer >= self.n_layers:
            raise ValueError(f"Layer index {layer} is out of bounds for n_layers={self.n_layers}")

        if cache_position < 0 or cache_position + k.shape[2] > self.max_seq_len:
            raise ValueError(
                f"Cache position {cache_position} with update length {k.shape[2]} "
                f"exceeds max_seq_len={self.max_seq_len}"
            )

        # update new keys and values to the existing cache tensor for the specified layer
        self.k_cache[layer][:, :, cache_position : cache_position + k.shape[2], :] = k
        self.v_cache[layer][:, :, cache_position : cache_position + v.shape[2], :] = v

        self.sequence_length = cache_position + k.shape[2]
        # (B, n_kv_heads, sequence_length, d_head)
        return (
            self.k_cache[layer][:, :, : self.sequence_length, :],
            self.v_cache[layer][:, :, : self.sequence_length, :],
        )
