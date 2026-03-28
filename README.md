# bob

From scratch GPT implementation with PyTorch. 


## Run

```bash
uv sync
uv run python scripts/generate.py --prompt "abcabc"
```

##  Architecture
- Multi-head self-attention with causal masking
- Rotary positional embeddings (RoPE)
- RMSNorm before attention and MLP
- Gated feed forward network with SwiGLU
- Tied input/output embeddings
- no biases in projections

## Todo 
- Training loop and evaluation
- Flexible sampling strategies in next token prediction
- Bytepair encoding tokenizer
- Ability to load external model weights
- KV cache to optimize self attention

