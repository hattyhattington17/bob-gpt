"""Bob the transformer model."""

import torch

from bob.config import ModelConfig
from bob.model.attention import SelfAttention
from bob.model.mlp import MLP
from bob.model.rmsnorm import RMSNorm
from bob.model.rope import RoPE


class Bob(torch.nn.Module):
    """GPT language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.rope = RoPE(config.d_head, config.max_seq_len, config.rope_theta)
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.norm = RMSNorm(config.d_model, config.norm_eps)

        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate logits for the next token.

        Args:
            x: Tensor of token ids with shape (B, T).

        Returns:
            Logits tensor with shape (B, T, vocab_size).
        """
        hidden_state = self.embeddings(x)  # (B, T, d_model)

        cos, sin = self.rope(hidden_state.shape[1])  # cos, sin each (T, d_head // 2)

        # transformer layers: (B, T, d_model) throughout
        for layer in self.layers:
            hidden_state = layer(hidden_state, cos, sin)

        hidden_state = self.norm(hidden_state)  # (B, T, d_model)
        logits = self.lm_head(hidden_state)  # (B, T, vocab_size)
        return logits


class TransformerBlock(torch.nn.Module):
    """Transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.norm1 = RMSNorm(config.d_model, config.norm_eps)
        self.norm2 = RMSNorm(config.d_model, config.norm_eps)
        self.self_attn = SelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and normalization with residual connections.

        Args:
            x: Hidden state, shape (B, T, d_model).
            cos: Cached cosine values, shape (T, d_head // 2).
            sin: Cached sine values, shape (T, d_head // 2).

        Returns:
            Output tensor, shape (B, T, d_model).
        """
        # first normalization
        u = self.norm1(x)  # (B, T, d_model)

        # self attention
        a = self.self_attn(u, cos, sin)  # (B, T, d_model)
        # residual connection
        y = x + a  # (B, T, d_model)

        # second normalization
        v = self.norm2(y)  # (B, T, d_model)
        # MLP
        m = self.mlp(v)

        # residual connection
        result = y + m
        return result
