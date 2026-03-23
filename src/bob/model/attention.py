"""Multihead self attention with causal masking."""

import torch

from bob.config import ModelConfig
from bob.model.rope import apply_rotary_emb


class SelfAttention(torch.nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head

        # W_q, W_k, W_v: weight shape (d_model, n_heads * d_head)
        # n_heads * d_head = d_model
        self.W_q = torch.nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.W_k = torch.nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.W_v = torch.nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)

        # W_out: weight shape (n_heads * d_head, d_model)
        self.W_out = torch.nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)
 
        # create a square matrix of ones keep the upper triangular part of the matrix (excluding 
        # diagonal) set the rest to zero and convert to booleans
        # (max_seq_len, max_seq_len)
        mask = torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool() 

        # register causal mask as buffer
        self.register_buffer("causal_mask", mask, persistent=False)  

        
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Compute multi-head self-attention with causal masking.

        Args:
            x: Normalized hidden state, shape (B, T, d_model).
            cos: Cached cosine values, shape (T, d_head // 2).
            sin: Cached sine values, shape (T, d_head // 2).

        Returns:
            Attention output, shape (B, T, d_model).
        """
        # project into query, key, and value tensors
        # x @ W_q: (B, T, d_model) @ (d_model, n_heads*d_head)
        q = self.W_q(x)  # (B, T, n_heads * d_head)
        k = self.W_k(x)  # (B, T, n_heads * d_head)
        v = self.W_v(x)  # (B, T, n_heads * d_head)

        # reshape into heads
        q = q.view(q.shape[0], q.shape[1], self.n_heads, self.d_head)  # (B, T, n_heads, d_head)
        k = k.view(k.shape[0], k.shape[1], self.n_heads, self.d_head)  # (B, T, n_heads, d_head)
        v = v.view(v.shape[0], v.shape[1], self.n_heads, self.d_head)  # (B, T, n_heads, d_head)

        # transpose head and sequence position axes (axis 1 and 2)
        q = q.transpose(1, 2)  # (B, n_heads, T, d_head)
        k = k.transpose(1, 2)  # (B, n_heads, T, d_head)
        v = v.transpose(1, 2)  # (B, n_heads, T, d_head)

        # apply RoPE
        q = apply_rotary_emb(q, cos, sin)  # (B, n_heads, T, d_head)
        k = apply_rotary_emb(k, cos, sin)  # (B, n_heads, T, d_head)

        # compute self attention scores
        # for each head h and position j, score_h(j, i) measures how relevant position i is to j
        # score_h(j,i) = Q[b,h,j] dot K[b,h,i] / sqrt(d_head)
        scores = q @ k.transpose(-2, -1) / (self.d_head**0.5)  # (B, n_heads, T, T)
        # in practice we would use
        # z = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        # apply causal mask
        T = scores.shape[-1]
      
        # set upper triangular scores to -inf so that after softmax they become zero
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))  # (B, n_heads, T, T)

        # compute attention weights with softmax on the last dimension
        attn_weights = torch.softmax(scores, dim=-1)  # (B, n_heads, T, T)

        # compute weighted sum of values: (B, n_heads, T, T) @ (B, n_heads, T, d_head)
        z = attn_weights @ v  # (B, n_heads, T, d_head)

        # transpose back
        z = z.transpose(1, 2)  # (B, T, n_heads, d_head)
        # concatenate heads
        z = z.reshape(z.shape[0], z.shape[1], self.n_heads * self.d_head)  # (B, T, n_heads * d_head)
        # apply learned output projection
        z = self.W_out(z)  # (B, T, d_model)

        return z
