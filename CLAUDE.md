# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                            # install deps
uv run pytest                      # run all tests
uv run pytest tests/test_foo.py    # run a single test file
uv run ruff check src              # lint
uv run ruff format src             # format
uv run mypy src                    # type check
uv run python scripts/generate.py --prompt "abcabc"   # run inference
```

Ruff uses Google-style docstrings (`pydocstyle.convention = "google"`), line length 100, and enforces `E, F, I, B, UP, D` rules (with `D104/D105/D107` ignored). Mypy is strict.

## Architecture

Bob is a from-scratch GPT built with PyTorch. The model is a standard decoder-only transformer with:
- Pre-norm (RMSNorm before attention and MLP, not after)
- RoPE positional embeddings (applied inside `SelfAttention`, not in `Bob`)
- SwiGLU MLP (`W_a` = candidate projection, `W_b` = gate; SiLU applied to `W_b`)
- Tied input/output embeddings (controlled by `ModelConfig.tie_embeddings`)
- No biases anywhere in projections

**Data flow:** `token_ids (B,T)` → `Embedding` → N × `TransformerBlock` → `RMSNorm` → `lm_head` → `logits (B,T,vocab_size)`

**Config** (`src/bob/config.py`): `ModelConfig` is a frozen dataclass loaded from YAML via `ModelConfig.from_yaml(path)`. `configs/nano.yaml` is a tiny 26-token char-level model for development/testing.

**Tokenizer** (`src/bob/tokenizer/`): Simple character-level tokenizer. `encode` returns `list[int]`, `decode` takes `list[int]`.

**Inference** (`src/bob/inference/`): `generate()` runs autoregressive generation with sliding-window context truncation. `greedy()` in `sampler.py` picks the argmax token.

**Scripts** (`scripts/generate.py`): Entry point. Detects MPS (Apple Silicon) and falls back to CPU.

## Planning docs

`planning/implementation.md` — Phase 1 (model architecture) plan and status.
`planning/training.md` — Phase 3 (training loop) plan: scaffold → dataset → LR schedule → checkpointing.
`notes/` — math reference notes for each module.

# Feedback loop
- after each change, run `uv run python scripts/generate.py --prompt "abcabc" ` to make sure the code still executes

# Gotchas
- pay close attention to precision in computations

# Git conventions
- 