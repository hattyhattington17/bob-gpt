# bob

From scratch GPT implementation with PyTorch. 


## Run

```bash
uv sync
uv run python scripts/generate.py --prompt "abcabc"
```

## Training

Download the Shakespeare dataset:

```bash
mkdir -p data/raw
curl -o data/raw/shakespeare.txt https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt
```

Then run the training loop:

```bash
uv run python scripts/train.py --config configs/nano.yaml
```

##  Architecture
- Multi-head self-attention with causal masking
- Rotary positional embeddings (RoPE)
- RMSNorm before attention and MLP
- Gated feed forward network with SwiGLU
- Tied input/output embeddings
- no biases in projections
 
