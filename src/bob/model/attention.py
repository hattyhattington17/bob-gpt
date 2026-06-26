"""Multihead self attention with causal masking."""

import torch

from bob.config import ModelConfig
from bob.model.rmsnorm import RMSNorm
from bob.model.rope import apply_rotary_emb


class SelfAttention(torch.nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.kv_heads = config.kv_heads

        # separate q and k norm modules with independent weights
        # annotate optional norm modules to satisfy type checkers
        self.q_norm: RMSNorm | None
        self.k_norm: RMSNorm | None
        if config.qk_norm:
            self.q_norm = RMSNorm(config.d_head, config.norm_eps)
            self.k_norm = RMSNorm(config.d_head, config.norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # (d_model, n_heads * d_head)
        self.W_q = torch.nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        # (d_model, kv_heads * d_head)
        self.W_k = torch.nn.Linear(config.d_model, config.kv_heads * config.d_head, bias=False)
        self.W_v = torch.nn.Linear(config.d_model, config.kv_heads * config.d_head, bias=False)

        # W_out: weight shape (n_heads * d_head, d_model)
        self.W_out = torch.nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute multi-head self-attention with causal masking.

        Args:
            x: Normalized hidden state, shape (B, T, d_model).
            cos: Cached cosine values, shape (T, d_head // 2).
            sin: Cached sine values, shape (T, d_head // 2).
            padding_mask: Optional padding mask, shape (B, T), where True
                indicates a padded token at some position in a batch

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

        # compute group size for GQA
        G = self.n_heads // self.kv_heads
        k = repeat_kv(k, G)
        v = repeat_kv(v, G)

        # torch.nn.functional.scaled_dot_product_attention computes scores, applies attention mask,
        # softmaxes into weights and computes weighted sum of values to produce output z
        # (B, n_heads, T, d_head)
        if padding_mask is not None:
            # if padding mask was supplied, combine with causal mask to construct attention mask
            # broadcast padding mask (B, T) to match shape of attention scores (B, n_heads, T, T)
            padding_mask = padding_mask[:, None, None, :]  # (B, 1, 1, T)
            # add causal masking to padding mask, set upper right triangle values to True (masked)
            causal_mask = torch.triu(
                torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), diagonal=1
            )  # (T, T)
            # broadcast causal mask (T, T) to match shape of attention scores (B, n_heads, T, T)
            causal_mask = causal_mask[None, None, :, :]  # (1, 1, T, T)
            # combine via OR - if either mask has True, the position is masked
            attn_mask = padding_mask | causal_mask
            attn_mask = torch.zeros(
                x.shape[0], 1, x.shape[1], x.shape[1], dtype=q.dtype, device=x.device
            ).masked_fill(attn_mask, torch.finfo(q.dtype).min)  # (B, 1, T, T)

            z = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask
            )  # (B, n_heads, T, d_head)
        else:
            z = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
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
