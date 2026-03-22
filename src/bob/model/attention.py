"""Multihead self attention with causal masking"""

import torch

from bob.config import ModelConfig


class SelfAttention(torch.nn.Module):
    """ Transformer block """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_head = config.d_head

        # W_q, W_k, W_v in R^{d_model × (n_heads * d_head)} 
        # n_heads * d_head = d_model 
        self.W_q = torch.nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.W_k = torch.nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)
        self.W_v = torch.nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)

        # W_out in R^{(n_heads * d_head) x d_model} 
        self.W_out = torch.nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input is a normalized hidden state tensor x in R^{B x T x d_model}

        # project into query, key, and value tensors in R^{B x T x (n_heads * d_head)}   
        # x @ W_q : (B x T x d_model) @ (d_model x (n_heads * d_head)) = Q in R^{B x T x (n_heads * d_head)}   
        Query = self.W_q(x)
        Key = self.W_k(x)
        Value = self.W_v(x)

        # reshape into heads Q, K, V  in R^{B x T x n_heads x d_head}
        Query = Query.view(Query.shape[0], Query.shape[1], self.config.n_heads, self.config.d_head)
        Key = Key.view(Key.shape[0], Key.shape[1], self.config.n_heads, self.config.d_head)
        Value = Value.view(Value.shape[0], Value.shape[1], self.config.n_heads, self.config.d_head)
       
        # transpose head and sequence position axes (axis 1 and 2) 
        # Q, K, V  in R^{B x n_heads x T x d_head}
        Query = Query.transpose(1, 2)
        Key = Key.transpose(1, 2)
        Value = Value.transpose(1, 2)

        # compute self attention scores
        # for each head h and position j, we have T scores for how much position i in [T] is relevant to position j 
        # score_h(j,i) = Q[b,h,j] dot K[b,h,i] / sqrt(d_head)
        # scores in R^{B x n_heads x T x T}
        scores = Query @ Key.transpose(3,2) / (self.config.d_head ** 0.5)
        # in practice we would use
        # z = torch.nn.functional.scaled_dot_product_attention(Query, Key, Value, is_causal=True)
        
        # apply causal mask
        T = scores.shape[-1]
        # create a T x T square matrix of ones 
        # keep only the upper triangular part of the matrix, including the diagonal,
        # set the rest to zero and convert to booleans
        mask = torch.triu(torch.ones(T, T,device=scores.device), diagonal=1).bool()
        # set upper triangular scores to -inf so that after softmax they become zero
        scores = scores.masked_fill(mask, float('-inf'))

        # compute attention weights with softmax on the last dimension
        # attn_weightsin R^{B x n_heads x T x T}
        attn_weights = torch.softmax(scores, dim=-1)

        # compute weighted sum of values
        # z in R^{B x n_heads x T x d_head}
        z = attn_weights @ Value

        z = z.transpose(1,2)
        # concatenate heads
        z = z.reshape(z.shape[0], z.shape[1], self.config.n_heads * self.config.d_head)
        # apply learned output projection
        z = self.W_out(z)
        
        # self attention output in R^{B x T x d_model}
        return z