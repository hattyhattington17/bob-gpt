"""Multihead self attention with causal masking."""

import torch

from bob.config import ModelConfig
from bob.inference.kv_cache import KVCache
from bob.model.rmsnorm import RMSNorm
from bob.model.rope import apply_rotary_emb


class SelfAttention(torch.nn.Module):
    """Multi-head self-attention module. Supports GQA."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.kv_heads = config.kv_heads

        # separate q and k RMSNorm modules with independent weights
        self.q_norm: RMSNorm | None
        self.k_norm: RMSNorm | None
        if config.qk_norm:
            self.q_norm = RMSNorm(config.d_head, config.norm_eps)
            self.k_norm = RMSNorm(config.d_head, config.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # W_q: d_model -> n_heads * d_head
        self.W_q = torch.nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        # W_k, W_v: d_model -> kv_heads * d_head
        self.W_k = torch.nn.Linear(config.d_model, config.kv_heads * config.d_head, bias=False)
        self.W_v = torch.nn.Linear(config.d_model, config.kv_heads * config.d_head, bias=False)

        # W_out: n_heads * d_head -> d_model
        self.W_out = torch.nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_index: int,
        padding_mask: torch.Tensor,
        past_seen_tokens: int = 0,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Compute multi-head self-attention with causal masking.

        Args:
            x: Normalized hidden state, shape (B, T, d_model).
            cos: Cached cosine values for each RoPE absolute position, shape (B, T, d_head // 2).
            sin: Cached sine values for each RoPE absolute position, shape (B, T, d_head // 2).
            layer_index: index of the layer in the transformer.
            padding_mask: Padding mask, shape (B, past_seen_tokens  + T)
            past_seen_tokens: positions cached, same for all seqs due to left padding
            kv_cache: Key-value cache for self attention.

        Returns:
            Attention output, shape (B, T, d_model).
        """
        # project into query, key, and value tensors
        q = self.W_q(x)  # (B, T, n_heads * d_head)
        k = self.W_k(x)  # (B, T, kv_heads * d_head)
        v = self.W_v(x)  # (B, T, kv_heads * d_head)

        # reshape and transpose into heads
        q = q.view(q.shape[0], q.shape[1], self.n_heads, self.d_head)  # (B, T, n_heads, d_head)
        k = k.view(k.shape[0], k.shape[1], self.kv_heads, self.d_head)  # (B, T, kv_heads, d_head)
        v = v.view(v.shape[0], v.shape[1], self.kv_heads, self.d_head)  # (B, T, kv_heads, d_head)
        q = q.transpose(1, 2)  # (B, n_heads, T, d_head)
        k = k.transpose(1, 2)  # (B, kv_heads, T, d_head)
        v = v.transpose(1, 2)  # (B, kv_heads, T, d_head)

        # qk norm
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # apply RoPE
        q = apply_rotary_emb(q, cos, sin)  # (B, n_heads, T, d_head)
        k = apply_rotary_emb(k, cos, sin)  # (B, kv_heads, T, d_head)

        # update KV cache w key and values for the input sequence
        # retrieve full cache for prev positions
        if kv_cache is not None:
            # kv_length += T
            k, v = kv_cache.update(
                layer_index, k, v, past_seen_tokens
            )  # (B, kv_heads, kv_length, d_head)

        # G = number of query heads sharing each key/value head
        G = self.n_heads // self.kv_heads
        # repeat key value heads G times each to match the shape of the query heads
        k = repeat_kv(k, G)  # (B, n_heads, kv_length, d_head)
        v = repeat_kv(v, G)  # (B, n_heads, kv_length, d_head)

        # build causal mask over full sequence kv_length
        # q is only over T positions, k/v are over kv_length positions
        kv_absolute_positions = torch.arange(k.shape[2], device=x.device)  # (kv_length)
        q_absolute_positions = torch.arange(q.shape[2], device=x.device) + past_seen_tokens  # (T)
        # true=attend for all q positions >= kv positions
        causal_mask = (
            q_absolute_positions.unsqueeze(1)  # (T, 1)
            >= kv_absolute_positions.unsqueeze(0)  # (1, kv_length)
        )  # (T, kv_length)
        # broadcast over batch and head dimensions
        causal_mask = causal_mask[None, None, :, :]  # (1, 1, T, kv_length)

        # combine causal mask with padding mask to construct attention mask
        padding_mask = padding_mask.bool()[:, None, None, :]  # (B, 1, 1, kv_length))
        attn_mask = causal_mask & padding_mask  # (B, 1, T, kv_length)

        # torch.nn.functional.scaled_dot_product_attention computes scores, applies attention mask,
        # softmaxes into weights and computes weighted sum of values to produce output z
        z = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )  # (B, n_heads, T, d_head)

        # (B, T, n_heads, d_head)
        z = z.transpose(1, 2)

        # (B, T, n_heads * d_head)
        z = z.reshape(z.shape[0], z.shape[1], self.n_heads * self.d_head)

        # apply learned output projection
        out: torch.Tensor = self.W_out(z)  # (B, T, d_model)
        return out


def repeat_kv(x: torch.Tensor, G: int) -> torch.Tensor:
    """Repeat key/value heads for GQA.

    Args:
        x: Key or value tensor of shape (B, n_kv_heads, T, d_head).
        G: Group size, number of query heads sharing a kv head,
            the number of times to repeat each key/value head.
    """
    # repeat each key/value head G times along the head dimension
    # (B, n_kv_heads, T, d_head) -> (B, n_kv_heads * G, T, d_head)
    return x.repeat_interleave(G, dim=1)
