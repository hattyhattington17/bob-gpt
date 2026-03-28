# Training Loop Design

## Goal

Train Bob on Bob-ified Shakespeare (all character names replaced with Bob1, Bob2, etc.) using a standard GPT training loop with cosine LR schedule, checkpointing, and periodic eval with sample generation.

---

## Preprocessing

**Script:** `scripts/bobify.py`

One-time script. Reads raw Shakespeare text, builds a deterministic character name → numbered Bob mapping (names sorted alphabetically so Bob1 is always the same character), replaces all occurrences, writes `data/input.txt`. The name list is hardcoded in the script. Output is kept locally (not committed to git).

---

## Config

**File:** `configs/nano.yaml`

Extended with a `training:` section alongside the existing `model:` section. One file per model — they're versioned together since a training run is always tied to a specific model shape.

```yaml
model:
  vocab_size: ...
  ...

training:
  data_path: data/input.txt
  train_split: 0.9
  batch_size: 32
  max_steps: 5000
  warmup_steps: 100
  learning_rate: 3.0e-4
  min_lr: 3.0e-5
  weight_decay: 0.1
  grad_clip: 1.0
  eval_interval: 500
  eval_steps: 50
  checkpoint_interval: 1000
  checkpoint_dir: checkpoints
```

**`src/bob/config.py`:** Add `TrainingConfig` frozen dataclass with `TrainingConfig.from_yaml(path)`. `ModelConfig.from_yaml` stays unchanged — both read their respective keys from the same YAML.

---

## File Structure

```
bob/
├── configs/
│   └── nano.yaml              ← extended with training: section
├── data/
│   └── input.txt              ← bobified shakespeare (not committed)
├── checkpoints/               ← saved checkpoints (not committed)
├── src/bob/
│   ├── config.py              ← add TrainingConfig
│   └── training/
│       ├── __init__.py
│       ├── dataset.py         ← CharDataset + build_dataloaders
│       ├── schedule.py        ← get_lr (cosine warmup)
│       ├── checkpoint.py      ← save/load checkpoint + vocab
│       └── trainer.py         ← train loop
└── scripts/
    ├── bobify.py              ← one-time preprocessing
    └── train.py               ← entry point
```

---

## Dataset (`src/bob/training/dataset.py`)

`CharDataset` takes a list of token IDs and `seq_len`. Each item returns `(x, y)` where `y` is `x` shifted by one token (next-token prediction).

`build_dataloaders` reads `data/input.txt`, builds `Tokenizer.from_text(text)`, splits into train/val by token count (`train_split`), wraps each split in a `CharDataset`, and returns `(train_loader, val_loader, tokenizer)`. The tokenizer is returned so it can be persisted to `vocab.json`.

---

## Checkpointing (`src/bob/training/checkpoint.py`)

Two responsibilities:

1. **Model checkpoints** — saves `{step, model_state_dict, optimizer_state_dict}` to `checkpoints/ckpt_XXXXXX.pt`. On resume, loads the latest checkpoint automatically by sorting filenames.
2. **Vocab** — saves/loads `checkpoints/vocab.json` (the sorted char list from the tokenizer). Written once at the start of training; read by inference scripts to reconstruct the tokenizer.

---

## LR Schedule (`src/bob/training/schedule.py`)

Single function `get_lr(step, warmup_steps, max_steps, max_lr, min_lr) -> float`:

- Linear warmup from 0 to `max_lr` over `warmup_steps`
- Cosine decay from `max_lr` to `min_lr` from `warmup_steps` to `max_steps`
- Returns `min_lr` for any step beyond `max_steps`

---

## Training Loop (`src/bob/training/trainer.py`)

`train(model_config, config, device)`:

1. Build dataloaders + tokenizer, save vocab
2. Assert `model_config.vocab_size == tokenizer.vocab_size` — fail loudly if not (nano.yaml must be updated to match the actual Shakespeare vocab size, ~65, after running `bobify.py`)
3. Load latest checkpoint if one exists (sets starting step)
3. Build optimizer with two param groups:
   - **Decay:** 2D projection weights (excludes names containing `embed` or `norm`)
   - **No decay:** everything else (embeddings, norm scale, 1D params)
4. Loop `max_steps` steps:
   - Pull next batch from cycling train DataLoader
   - Forward → cross-entropy loss
   - Zero grad → backward → clip grads (`grad_clip`) → set LR → optimizer step
5. Every `eval_interval` steps:
   - Compute val loss under `@torch.inference_mode()`
   - Generate a short sample using existing `generate()` + `greedy()`
   - Log: `step | train_loss | val_loss | lr | sample`
6. Every `checkpoint_interval` steps: save checkpoint

---

## Entry Point (`scripts/train.py`)

Parses `--config` path, loads `ModelConfig` and `TrainingConfig` from the same YAML, detects device (MPS → CPU fallback), calls `train()`.

```bash
uv run python scripts/train.py --config configs/nano.yaml
```

---

## What This Produces

Full training loop: bobify text → tokenize → train with cosine LR → log train/val loss + LR + generated sample → checkpoint periodically. Resumable from latest checkpoint on restart. Vocab saved alongside checkpoints for use by inference.
