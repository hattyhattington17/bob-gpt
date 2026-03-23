"""Multihead self attention with causal masking"""

import torch

from bob.model.rope import apply_rotary_emb


class SelfAttention(torch.nn.Module):
    """Multi head self attention module"""

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head

        # W_q, W_k, W_v in R^{d_model × (n_heads * d_head)}
        # n_heads * d_head = d_model
        self.W_q = torch.nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_k = torch.nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_v = torch.nn.Linear(d_model, n_heads * d_head, bias=False)

        # W_out in R^{(n_heads * d_head) x d_model}
        self.W_out = torch.nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        input is a normalized hidden state tensor x in R^{B x T x d_model} and
        cos, sin in R^{T, d_head / 2} - cached rotary embedding values for the current sequence length
        """

        # project into query, key, and value tensors in R^{B x T x (n_heads * d_head)}
        # x @ W_q : (B x T x d_model) @ (d_model x (n_heads * d_head)) = Q in R^{B x T x (n_heads * d_head)}
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # reshape into heads Q, K, V  in R^{B x T x n_heads x d_head}
        q = q.view(q.shape[0], q.shape[1], self.n_heads, self.d_head)
        k = k.view(k.shape[0], k.shape[1], self.n_heads, self.d_head)
        v = v.view(v.shape[0], v.shape[1], self.n_heads, self.d_head)

        # transpose head and sequence position axes (axis 1 and 2)
        # Q, K, V  in R^{B x n_heads x T x d_head}
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # apply RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # compute self attention scores
        # for each head h and position j, we have T scores for how much position i in [T] is relevant to position j
        # score_h(j,i) = Q[b,h,j] dot K[b,h,i] / sqrt(d_head)
        # scores in R^{B x n_heads x T x T}
        scores = q @ k.transpose(-2, -1) / (self.d_head**0.5)
        # in practice we would use
        # z = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        # apply causal mask
        T = scores.shape[-1]
        # create a T x T square matrix of ones
        # keep only the upper triangular part of the matrix, exclude the diagonal
        # set the rest to zero and convert to booleans
        # todo: register in cache as a buffer, same for all heads
        mask = torch.triu(torch.ones(T, T, device=scores.device), diagonal=1).bool()
        # set upper triangular scores to -inf so that after softmax they become zero
        scores = scores.masked_fill(mask, float("-inf"))

        # compute attention weights with softmax on the last dimension
        # attn_weightsin R^{B x n_heads x T x T}
        attn_weights = torch.softmax(scores, dim=-1)

        # compute weighted sum of values
        # z in R^{B x n_heads x T x d_head}
        z = attn_weights @ v

        z = z.transpose(1, 2)
        # concatenate heads
        z = z.reshape(z.shape[0], z.shape[1], self.n_heads * self.d_head)
        # apply learned output projection
        z = self.W_out(z)

        # self attention output in R^{B x T x d_model}
        return z
