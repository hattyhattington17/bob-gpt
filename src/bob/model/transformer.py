"""Bob the transformer model."""

import torch

from bob.config import ModelConfig
from bob.inference.kv_cache import KVCache
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

        # lm_head: d_model -> vocab_size
        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Compute next token logits over vocabulary for all positions in T-length input sequence.

        Args:
            input_ids: Tensor of input token ids with shape (B, T).
            padding_mask: Mask marking padding token positions as 0 (B, past_seen_tokens + T).
                Sequences must be left padded to the longest sequence in the batch
            kv_cache: Key-value cache for self attention.
                past_seen_tokens is the number of positions already cached
                the full sequence length is past_seen_tokens + T (must be <= max_seq_len)

        Returns:
            Logits tensor with shape (B, T, vocab_size).
        """
        # embedding layer
        hidden_state: torch.Tensor = self.embeddings(input_ids)  # (B, T, d_model)

        # track starting cache position before we start appending at each layer
        # pass to the layers so they know where to append
        past_seen_tokens = kv_cache.kv_length if kv_cache is not None else 0

        # Compute absolute position of each token in the input sequence for RoPE
        # get padding per sequence from padding mask (0 = pad, 1 = real)
        # ex: batch size 1, 2 padding positions [[0, 0, 1, 1, 1]]
        # cumsum along the position dimension [[0, 0, 1, 2, 3]]
        # subtract 1 for indexing [[-1, -1, 0, 1, 2]]
        # clamp to 0 to remove -1s at padding positions [[0, 0, 0, 1, 2]]
        token_positions = (padding_mask.cumsum(-1) - 1).clamp(min=0)  # (B, past_seen_tokens + T)

        # take absolute positions for the current model input sequence
        # absolute positions for just the input sequence length
        T = hidden_state.shape[1]
        rope_positions = token_positions[:, -T:]  # (B, T)

        # load RoPE rotation cache for the absolute positions being processed
        cos, sin = self.rope(rope_positions)  # (B, T, d_head // 2)

        # transformer layers: (B, T, d_model) throughout
        for index, layer in enumerate(self.layers):
            hidden_state = layer(
                hidden_state,
                cos,
                sin,
                index,
                past_seen_tokens=past_seen_tokens,
                kv_cache=kv_cache,
                padding_mask=padding_mask,
            )

        hidden_state = self.norm(hidden_state)  # (B, T, d_model)
        out: torch.Tensor = self.lm_head(hidden_state)  # (B, T, vocab_size)
        return out


class TransformerBlock(torch.nn.Module):
    """Transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.norm1 = RMSNorm(config.d_model, config.norm_eps)
        self.norm2 = RMSNorm(config.d_model, config.norm_eps)
        self.self_attn = SelfAttention(config)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_index: int,
        padding_mask: torch.Tensor,
        past_seen_tokens: int = 0,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """Transformer layer applies self attention and MLP.

        Args:
            hidden_states: Hidden state, shape (B, T, d_model).
            cos: Cached cosine values for each RoPE position, shape (B, T, d_head // 2).
            sin: Cached sine values for each RoPE position, shape (B, T, d_head // 2).
            layer_index: index of the layer in the transformer.
            padding_mask: Padding mask, shape (B, past_seen_tokens + T)
            past_seen_tokens: number of positions already cached, same for all sequences.
            kv_cache: Key-value cache for self attention.

        Returns:
            Output tensor, shape (B, T, d_model).
        """
        residual_stream: torch.Tensor = hidden_states  # (B, T, d_model)

        # attention subblock
        hidden_states = self.norm1(hidden_states)  # (B, T, d_model)
        hidden_states = self.self_attn(
            hidden_states,
            cos,
            sin,
            layer_index,
            past_seen_tokens=past_seen_tokens,
            kv_cache=kv_cache,
            padding_mask=padding_mask,
        )
        residual_stream = residual_stream + hidden_states

        # MLP subblock
        hidden_states = self.norm2(residual_stream)
        hidden_states = self.mlp(hidden_states)
        residual_stream = residual_stream + hidden_states

        return residual_stream  # (B, T, d_model)
