# Training Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full training loop for Bob on bobified Shakespeare text, with cosine LR schedule, checkpointing, and periodic eval with sample generation.

**Architecture:** Training infrastructure lives in `src/bob/training/` (dataset, schedule, checkpoint, trainer). A one-time `scripts/bobify.py` preprocessing script produces `data/input.txt`. `scripts/train.py` is the entry point. `TrainingConfig` is added to `src/bob/config.py` alongside the existing `ModelConfig`.

**Tech Stack:** PyTorch (DataLoader, AdamW, cross_entropy), uv/pytest for testing, PyYAML for config.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/bob/config.py` | Modify | Add `TrainingConfig` frozen dataclass |
| `configs/nano.yaml` | Modify | Add `training:` section |
| `src/bob/tokenizer/tokenizer.py` | Modify | Add `chars` property |
| `src/bob/training/__init__.py` | Create | Package marker |
| `src/bob/training/dataset.py` | Create | `CharDataset` + `build_dataloaders` |
| `src/bob/training/schedule.py` | Create | `get_lr` cosine warmup function |
| `src/bob/training/checkpoint.py` | Create | Save/load checkpoints + vocab |
| `src/bob/training/trainer.py` | Create | `train()` loop |
| `scripts/bobify.py` | Create | One-time Shakespeare preprocessing |
| `scripts/train.py` | Create | CLI entry point |
| `tests/training/__init__.py` | Create | Package marker |
| `tests/training/test_config.py` | Create | TrainingConfig tests |
| `tests/training/test_dataset.py` | Create | CharDataset + build_dataloaders tests |
| `tests/training/test_schedule.py` | Create | get_lr tests |
| `tests/training/test_checkpoint.py` | Create | Checkpoint + vocab tests |
| `tests/training/test_trainer.py` | Create | Smoke test for train() |

---

## Task 1: TrainingConfig

**Files:**
- Modify: `src/bob/config.py`
- Modify: `configs/nano.yaml`
- Create: `tests/training/__init__.py`
- Create: `tests/training/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_config.py
"""Tests for TrainingConfig."""

from pathlib import Path

import pytest

from bob.config import TrainingConfig


def test_training_config_from_yaml(tmp_path: Path) -> None:
    yaml_content = """
model:
  vocab_size: 26
  d_model: 48
  n_heads: 3
  n_layers: 3
  d_ff: 192
  max_seq_len: 10

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
"""
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    config = TrainingConfig.from_yaml(config_file)

    assert config.data_path == "data/input.txt"
    assert config.train_split == 0.9
    assert config.batch_size == 32
    assert config.max_steps == 5000
    assert config.warmup_steps == 100
    assert config.learning_rate == pytest.approx(3e-4)
    assert config.min_lr == pytest.approx(3e-5)
    assert config.weight_decay == pytest.approx(0.1)
    assert config.grad_clip == pytest.approx(1.0)
    assert config.eval_interval == 500
    assert config.eval_steps == 50
    assert config.checkpoint_interval == 1000
    assert config.checkpoint_dir == "checkpoints"


def test_training_config_is_frozen(tmp_path: Path) -> None:
    yaml_content = """
model:
  vocab_size: 26
  d_model: 48
  n_heads: 3
  n_layers: 3
  d_ff: 192
  max_seq_len: 10

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
"""
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)
    config = TrainingConfig.from_yaml(config_file)

    with pytest.raises(Exception):
        config.batch_size = 64  # type: ignore[misc]
```

- [ ] **Step 2: Create test package marker and run tests to verify they fail**

```bash
touch tests/training/__init__.py
```

Run: `uv run pytest tests/training/test_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'TrainingConfig' from 'bob.config'`

- [ ] **Step 3: Add TrainingConfig to config.py**

Full updated `src/bob/config.py`:

```python
"""Load and validate model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for the model architecture."""

    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_seq_len: int
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True

    @property
    def d_head(self) -> int:
        """Return the dimension of each attention head."""
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.d_head % 2 != 0:
            raise ValueError(f"d_head ({self.d_head}) must be even for RoPE")

    @classmethod
    def from_yaml(cls, path: Path) -> ModelConfig:
        """Load model config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw["model"])


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable configuration for the training loop."""

    data_path: str
    train_split: float
    batch_size: int
    max_steps: int
    warmup_steps: int
    learning_rate: float
    min_lr: float
    weight_decay: float
    grad_clip: float
    eval_interval: int
    eval_steps: int
    checkpoint_interval: int
    checkpoint_dir: str

    @classmethod
    def from_yaml(cls, path: Path) -> TrainingConfig:
        """Load training config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw["training"])
```

- [ ] **Step 4: Add training section to nano.yaml**

Full updated `configs/nano.yaml`:

```yaml
model:
  vocab_size: 26
  d_model: 48
  n_heads: 3
  n_layers: 3
  d_ff: 192
  max_seq_len: 10
  rope_theta: 10000.0
  norm_eps: 1.0e-6
  tie_embeddings: true

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

Note: `vocab_size` will be updated in Task 6 after running `bobify.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/training/test_config.py -v`
Expected: 2 passed

- [ ] **Step 6: Verify generate script still works**

Run: `uv run python scripts/generate.py --prompt "abcabc"`
Expected: prints `you:  abcabc` and `bob:  ...` without errors

- [ ] **Step 7: Commit**

```bash
git add src/bob/config.py configs/nano.yaml tests/training/__init__.py tests/training/test_config.py
git commit -m "feat: add TrainingConfig frozen dataclass"
```

---

## Task 2: Dataset

**Files:**
- Create: `src/bob/training/__init__.py`
- Create: `src/bob/training/dataset.py`
- Modify: `src/bob/tokenizer/tokenizer.py` (add `chars` property)
- Create: `tests/training/test_dataset.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_dataset.py
"""Tests for CharDataset and build_dataloaders."""

from pathlib import Path

import torch

from bob.training.dataset import CharDataset, build_dataloaders


def test_char_dataset_len() -> None:
    ids = list(range(10))
    ds = CharDataset(ids, seq_len=3)
    assert len(ds) == 7  # 10 - 3


def test_char_dataset_getitem_x_y_shift() -> None:
    ids = [0, 1, 2, 3, 4]
    ds = CharDataset(ids, seq_len=3)
    x, y = ds[0]
    assert x.tolist() == [0, 1, 2]
    assert y.tolist() == [1, 2, 3]


def test_char_dataset_getitem_last() -> None:
    ids = [0, 1, 2, 3, 4]
    ds = CharDataset(ids, seq_len=3)
    x, y = ds[1]
    assert x.tolist() == [1, 2, 3]
    assert y.tolist() == [2, 3, 4]


def test_char_dataset_tensor_dtype() -> None:
    ds = CharDataset([0, 1, 2, 3], seq_len=2)
    x, y = ds[0]
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_build_dataloaders_returns_tokenizer(tmp_path: Path) -> None:
    text = "hello world " * 100
    data_file = tmp_path / "input.txt"
    data_file.write_text(text)

    train_loader, val_loader, tokenizer = build_dataloaders(
        str(data_file), train_split=0.9, seq_len=8, batch_size=4
    )

    assert tokenizer.vocab_size == len(set(text))


def test_build_dataloaders_split_sizes(tmp_path: Path) -> None:
    # 1000 chars → 900 train tokens, 100 val tokens
    text = "abcde" * 200  # 1000 chars, vocab size 5
    data_file = tmp_path / "input.txt"
    data_file.write_text(text)

    train_loader, val_loader, tokenizer = build_dataloaders(
        str(data_file), train_split=0.9, seq_len=4, batch_size=8
    )

    # train split has 900 tokens → 896 samples (900 - 4)
    assert len(train_loader.dataset) == 896  # type: ignore[arg-type]
    # val split has 100 tokens → 96 samples (100 - 4)
    assert len(val_loader.dataset) == 96  # type: ignore[arg-type]


def test_build_dataloaders_batch_shape(tmp_path: Path) -> None:
    text = "abcdefgh" * 50
    data_file = tmp_path / "input.txt"
    data_file.write_text(text)

    train_loader, _, _ = build_dataloaders(
        str(data_file), train_split=0.9, seq_len=8, batch_size=4
    )

    x, y = next(iter(train_loader))
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/training/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bob.training'`

- [ ] **Step 3: Create package marker and dataset module**

Create `src/bob/training/__init__.py` (empty file):
```python
```

Create `src/bob/training/dataset.py`:

```python
"""Character-level dataset and dataloader construction."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from bob.tokenizer.tokenizer import Tokenizer


class CharDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset of (x, y) pairs for next-token prediction.

    Each item is a sequence x of length seq_len and target y = x shifted right by one.
    """

    def __init__(self, token_ids: list[int], seq_len: int) -> None:
        """Initialize with token IDs and sequence length."""
        self._ids = token_ids
        self._seq_len = seq_len

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self._ids) - self._seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y) pair at index idx."""
        chunk = self._ids[idx : idx + self._seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def build_dataloaders(
    data_path: str,
    train_split: float,
    seq_len: int,
    batch_size: int,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]], Tokenizer]:
    """Read text, tokenize, split into train/val, return dataloaders and tokenizer.

    Args:
        data_path: Path to text file.
        train_split: Fraction of tokens used for training (e.g. 0.9).
        seq_len: Sequence length for each sample.
        batch_size: Batch size for both loaders.

    Returns:
        Tuple of (train_loader, val_loader, tokenizer).
    """
    text = Path(data_path).read_text()
    tokenizer = Tokenizer.from_text(text)
    ids = tokenizer.encode(text)

    split = int(len(ids) * train_split)
    train_ds = CharDataset(ids[:split], seq_len)
    val_ds = CharDataset(ids[split:], seq_len)

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, tokenizer
```

- [ ] **Step 4: Add `chars` property to Tokenizer**

Edit `src/bob/tokenizer/tokenizer.py` — add this property after `vocab_size`:

```python
    @property
    def chars(self) -> list[str]:
        """Return the sorted list of characters in the vocabulary."""
        return list(self._id_to_char)
```

Full updated `src/bob/tokenizer/tokenizer.py`:

```python
"""Character-level tokenizer mapping characters to IDs."""


class Tokenizer:
    """Character-level tokenizer mapping characters to integer IDs."""

    def __init__(self, chars: list[str]) -> None:
        """Initialize with list of all characters in the vocabulary."""
        self._char_to_id = {ch: i for i, ch in enumerate(chars)}
        self._id_to_char = list(chars)  # can index directly by ID to get char

    @classmethod
    def from_text(cls, text: str) -> "Tokenizer":
        """Initialize the tokenizer with a vocab of all unique characters in the string."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self._char_to_id)

    @property
    def chars(self) -> list[str]:
        """Return the sorted list of characters in the vocabulary."""
        return list(self._id_to_char)

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs."""
        for ch in text:
            if ch not in self._char_to_id:
                raise ValueError(f"Character '{ch}' not in tokenizer vocabulary")
        return [self._char_to_id[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        """Decode list of token IDs to text."""
        return "".join(self._id_to_char[i] for i in token_ids)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/training/test_dataset.py -v`
Expected: 7 passed

- [ ] **Step 6: Run all tests to check nothing is broken**

Run: `uv run pytest -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/bob/training/__init__.py src/bob/training/dataset.py src/bob/tokenizer/tokenizer.py tests/training/test_dataset.py
git commit -m "feat: add CharDataset, build_dataloaders, Tokenizer.chars"
```

---

## Task 3: LR Schedule

**Files:**
- Create: `src/bob/training/schedule.py`
- Create: `tests/training/test_schedule.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_schedule.py
"""Tests for cosine LR schedule with linear warmup."""

import math

from bob.training.schedule import get_lr


def test_get_lr_at_step_zero() -> None:
    lr = get_lr(step=0, warmup_steps=100, max_steps=1000, max_lr=3e-4, min_lr=3e-5)
    assert lr == 0.0


def test_get_lr_warmup_midpoint() -> None:
    # at step 50 of 100 warmup, lr should be half of max_lr
    lr = get_lr(step=50, warmup_steps=100, max_steps=1000, max_lr=3e-4, min_lr=3e-5)
    assert abs(lr - 1.5e-4) < 1e-12


def test_get_lr_end_of_warmup() -> None:
    # at the last warmup step, lr should equal max_lr
    lr = get_lr(step=100, warmup_steps=100, max_steps=1000, max_lr=3e-4, min_lr=3e-5)
    assert abs(lr - 3e-4) < 1e-12


def test_get_lr_cosine_midpoint() -> None:
    # midpoint of cosine decay (step=550, warmup=100, max=1000):
    # progress = (550 - 100) / (1000 - 100) = 0.5
    # cos(pi * 0.5) = 0 → lr = min_lr + 0.5*(max_lr - min_lr)*(1+0) = midpoint
    lr = get_lr(step=550, warmup_steps=100, max_steps=1000, max_lr=3e-4, min_lr=3e-5)
    expected = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1.0 + math.cos(math.pi * 0.5))
    assert abs(lr - expected) < 1e-12


def test_get_lr_at_max_steps() -> None:
    # at max_steps, lr should equal min_lr
    lr = get_lr(step=1000, warmup_steps=100, max_steps=1000, max_lr=3e-4, min_lr=3e-5)
    assert abs(lr - 3e-5) < 1e-12


def test_get_lr_beyond_max_steps() -> None:
    # beyond max_steps, lr stays at min_lr
    lr = get_lr(step=9999, warmup_steps=100, max_steps=1000, max_lr=3e-4, min_lr=3e-5)
    assert lr == 3e-5


def test_get_lr_is_monotone_decreasing_after_warmup() -> None:
    steps = list(range(100, 1001, 50))
    lrs = [get_lr(s, warmup_steps=100, max_steps=1000, max_lr=3e-4, min_lr=3e-5) for s in steps]
    for a, b in zip(lrs, lrs[1:]):
        assert a >= b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/training/test_schedule.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bob.training.schedule'`

- [ ] **Step 3: Implement get_lr**

Create `src/bob/training/schedule.py`:

```python
"""Cosine learning rate schedule with linear warmup."""

from __future__ import annotations

import math


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Compute learning rate at a given step.

    Linear warmup from 0 to max_lr over warmup_steps, then cosine decay
    from max_lr to min_lr over the remaining steps. Returns min_lr beyond max_steps.

    Args:
        step: Current training step (0-indexed).
        warmup_steps: Number of warmup steps.
        max_steps: Total training steps.
        max_lr: Peak learning rate.
        min_lr: Minimum learning rate (floor of cosine decay).

    Returns:
        Learning rate for this step.
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/training/test_schedule.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/bob/training/schedule.py tests/training/test_schedule.py
git commit -m "feat: add cosine LR schedule with linear warmup"
```

---

## Task 4: Checkpointing

**Files:**
- Create: `src/bob/training/checkpoint.py`
- Create: `tests/training/test_checkpoint.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_checkpoint.py
"""Tests for checkpoint save/load and vocab persistence."""

from pathlib import Path

import torch

from bob.training.checkpoint import (
    load_latest_checkpoint,
    load_vocab,
    save_checkpoint,
    save_vocab,
)


def _make_model_and_optimizer() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = torch.nn.Linear(4, 4, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def test_save_checkpoint_creates_file(tmp_path: Path) -> None:
    model, optimizer = _make_model_and_optimizer()
    save_checkpoint(42, model, optimizer, str(tmp_path))
    assert (tmp_path / "ckpt_000042.pt").exists()


def test_load_latest_checkpoint_restores_step(tmp_path: Path) -> None:
    model, optimizer = _make_model_and_optimizer()
    save_checkpoint(42, model, optimizer, str(tmp_path))

    model2, optimizer2 = _make_model_and_optimizer()
    step = load_latest_checkpoint(model2, optimizer2, str(tmp_path))
    assert step == 42


def test_load_latest_checkpoint_restores_weights(tmp_path: Path) -> None:
    model, optimizer = _make_model_and_optimizer()
    # set known weights
    with torch.no_grad():
        model.weight.fill_(1.23)
    save_checkpoint(10, model, optimizer, str(tmp_path))

    model2, optimizer2 = _make_model_and_optimizer()
    with torch.no_grad():
        model2.weight.fill_(0.0)
    load_latest_checkpoint(model2, optimizer2, str(tmp_path))

    assert torch.allclose(model2.weight, torch.full((4, 4), 1.23))


def test_load_latest_checkpoint_picks_latest(tmp_path: Path) -> None:
    model, optimizer = _make_model_and_optimizer()
    save_checkpoint(100, model, optimizer, str(tmp_path))
    save_checkpoint(200, model, optimizer, str(tmp_path))

    model2, optimizer2 = _make_model_and_optimizer()
    step = load_latest_checkpoint(model2, optimizer2, str(tmp_path))
    assert step == 200


def test_load_latest_checkpoint_no_checkpoint(tmp_path: Path) -> None:
    model, optimizer = _make_model_and_optimizer()
    step = load_latest_checkpoint(model, optimizer, str(tmp_path))
    assert step == 0


def test_load_latest_checkpoint_dir_missing(tmp_path: Path) -> None:
    model, optimizer = _make_model_and_optimizer()
    missing_dir = str(tmp_path / "nonexistent")
    step = load_latest_checkpoint(model, optimizer, missing_dir)
    assert step == 0


def test_save_and_load_vocab(tmp_path: Path) -> None:
    chars = ["!", " ", "a", "b", "c"]
    save_vocab(chars, str(tmp_path))
    assert (tmp_path / "vocab.json").exists()

    loaded = load_vocab(str(tmp_path))
    assert loaded == chars
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/training/test_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bob.training.checkpoint'`

- [ ] **Step 3: Implement checkpoint module**

Create `src/bob/training/checkpoint.py`:

```python
"""Save and load model checkpoints and vocabulary."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def save_checkpoint(
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
) -> None:
    """Save model and optimizer state to a numbered checkpoint file.

    Args:
        step: Current training step (used to name the file).
        model: Model whose state_dict to save.
        optimizer: Optimizer whose state_dict to save.
        checkpoint_dir: Directory to write the checkpoint to.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir) / f"ckpt_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_latest_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
) -> int:
    """Load the most recent checkpoint if one exists.

    Args:
        model: Model to load state into.
        optimizer: Optimizer to load state into.
        checkpoint_dir: Directory to search for checkpoints.

    Returns:
        The step stored in the checkpoint, or 0 if no checkpoint found.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return 0
    checkpoints = sorted(ckpt_dir.glob("ckpt_*.pt"))
    if not checkpoints:
        return 0
    # weights_only=False required: optimizer state dicts contain complex objects
    # that the restricted unpickler rejects in PyTorch >= 2.6. We own these files.
    ckpt = torch.load(checkpoints[-1], weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt["step"])


def save_vocab(chars: list[str], checkpoint_dir: str) -> None:
    """Persist the tokenizer character list to vocab.json.

    Args:
        chars: Sorted list of characters from the tokenizer.
        checkpoint_dir: Directory to write vocab.json to.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir) / "vocab.json"
    path.write_text(json.dumps(chars))


def load_vocab(checkpoint_dir: str) -> list[str]:
    """Load the character list from vocab.json.

    Args:
        checkpoint_dir: Directory containing vocab.json.

    Returns:
        Sorted list of characters.
    """
    path = Path(checkpoint_dir) / "vocab.json"
    return json.loads(path.read_text())  # type: ignore[no-any-return]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/training/test_checkpoint.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/bob/training/checkpoint.py tests/training/test_checkpoint.py
git commit -m "feat: add checkpoint save/load and vocab persistence"
```

---

## Task 5: Training Loop

**Files:**
- Create: `src/bob/training/trainer.py`
- Create: `tests/training/test_trainer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_trainer.py
"""Smoke test for the training loop."""

from pathlib import Path

from bob.config import ModelConfig, TrainingConfig
from bob.training.trainer import train


def test_train_runs_and_saves_checkpoint(tmp_path: Path) -> None:
    # synthetic text with a small vocab
    text = "hello world foo bar baz " * 100
    data_file = tmp_path / "input.txt"
    data_file.write_text(text)

    vocab_size = len(set(text))  # unique chars in the text

    model_config = ModelConfig(
        vocab_size=vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        max_seq_len=8,
    )
    training_config = TrainingConfig(
        data_path=str(data_file),
        train_split=0.9,
        batch_size=4,
        max_steps=6,
        warmup_steps=2,
        learning_rate=1e-3,
        min_lr=1e-4,
        weight_decay=0.1,
        grad_clip=1.0,
        eval_interval=6,
        eval_steps=2,
        checkpoint_interval=6,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    train(model_config, training_config, device="cpu")

    ckpt_dir = tmp_path / "checkpoints"
    assert any(ckpt_dir.glob("ckpt_*.pt")), "expected at least one checkpoint"
    assert (ckpt_dir / "vocab.json").exists(), "expected vocab.json"


def test_train_resumes_from_checkpoint(tmp_path: Path) -> None:
    text = "abcdefgh " * 100
    data_file = tmp_path / "input.txt"
    data_file.write_text(text)

    vocab_size = len(set(text))

    model_config = ModelConfig(
        vocab_size=vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        max_seq_len=8,
    )
    training_config = TrainingConfig(
        data_path=str(data_file),
        train_split=0.9,
        batch_size=4,
        max_steps=4,
        warmup_steps=1,
        learning_rate=1e-3,
        min_lr=1e-4,
        weight_decay=0.1,
        grad_clip=1.0,
        eval_interval=10,
        eval_steps=2,
        checkpoint_interval=4,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    # first run: trains to step 4, saves checkpoint
    train(model_config, training_config, device="cpu")
    assert any((tmp_path / "checkpoints").glob("ckpt_*.pt"))

    # second run with max_steps=4: resumes from step 4, loop body runs 0 times
    train(model_config, training_config, device="cpu")
    checkpoints = sorted((tmp_path / "checkpoints").glob("ckpt_*.pt"))
    # only one checkpoint should exist (step 4 from first run; second run does nothing)
    assert len(checkpoints) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/training/test_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bob.training.trainer'`

- [ ] **Step 3: Implement trainer**

Create `src/bob/training/trainer.py`:

```python
"""Training loop for Bob."""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

from bob.config import ModelConfig, TrainingConfig
from bob.inference.generate import generate
from bob.model.transformer import Bob
from bob.tokenizer.tokenizer import Tokenizer
from bob.training.checkpoint import load_latest_checkpoint, save_checkpoint, save_vocab
from bob.training.dataset import DataLoader, build_dataloaders
from bob.training.schedule import get_lr


def train(model_config: ModelConfig, config: TrainingConfig, device: str) -> None:
    """Run the training loop.

    Builds dataloaders, saves vocab, constructs model + optimizer, resumes from
    the latest checkpoint if one exists, then trains for config.max_steps steps.
    Logs train/val loss and a generated sample every eval_interval steps.
    Saves a checkpoint every checkpoint_interval steps.

    Args:
        model_config: Model architecture config.
        config: Training hyperparameters.
        device: Torch device string, e.g. "cpu" or "mps".
    """
    train_loader, val_loader, tokenizer = build_dataloaders(
        config.data_path, config.train_split, model_config.max_seq_len, config.batch_size
    )
    save_vocab(tokenizer.chars, config.checkpoint_dir)

    assert model_config.vocab_size == tokenizer.vocab_size, (
        f"vocab_size in config ({model_config.vocab_size}) != tokenizer vocab_size "
        f"({tokenizer.vocab_size}). Update nano.yaml vocab_size to match."
    )

    model = Bob(model_config).to(device)

    decay_params = [
        p
        for n, p in model.named_parameters()
        if p.ndim >= 2 and "embed" not in n and "norm" not in n
    ]
    no_decay_params = [
        p
        for n, p in model.named_parameters()
        if not (p.ndim >= 2 and "embed" not in n and "norm" not in n)
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )

    start_step = load_latest_checkpoint(model, optimizer, config.checkpoint_dir)
    train_iter = itertools.cycle(train_loader)

    for step in range(start_step, config.max_steps):
        model.train()
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        logits = model(x)  # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, model_config.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        lr = get_lr(step, config.warmup_steps, config.max_steps, config.learning_rate, config.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()

        if (step + 1) % config.eval_interval == 0:
            val_loss = _eval(model, val_loader, config.eval_steps, model_config.vocab_size, device)
            sample = _sample(model, tokenizer, val_loader, model_config, device)
            print(
                f"step {step + 1:5d} | train_loss {loss.item():.4f} | "
                f"val_loss {val_loss:.4f} | lr {lr:.2e} | {sample!r}"
            )

        if (step + 1) % config.checkpoint_interval == 0:
            save_checkpoint(step + 1, model, optimizer, config.checkpoint_dir)


def _eval(
    model: Bob,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    eval_steps: int,
    vocab_size: int,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.inference_mode():
        for x, y in val_loader:
            if count >= eval_steps:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
            count += 1
    return total_loss / count if count > 0 else 0.0


def _sample(
    model: Bob,
    tokenizer: Tokenizer,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model_config: ModelConfig,
    device: str,
) -> str:
    model.eval()
    x, _ = next(iter(val_loader))
    prompt_ids = x[0, :3].tolist()
    output_ids = generate(
        model, prompt_ids, max_new_tokens=40, max_seq_len=model_config.max_seq_len, device=device
    )
    return tokenizer.decode(output_ids)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/training/test_trainer.py -v`
Expected: 2 passed (these run actual training steps so may take ~10 seconds)

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/bob/training/trainer.py tests/training/test_trainer.py
git commit -m "feat: add training loop with cosine LR, eval, and checkpointing"
```

---

## Task 6: Bobify Script + Update nano.yaml

**Files:**
- Create: `scripts/bobify.py`

- [ ] **Step 1: Create bobify.py**

Create `scripts/bobify.py`:

```python
"""One-time preprocessing: replace Shakespeare character names with Bob1, Bob2, etc.

Usage:
    uv run python scripts/bobify.py --input data/raw/shakespeare.txt

Download raw Shakespeare text from Project Gutenberg:
    curl -o data/raw/shakespeare.txt https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt

Output is written to data/input.txt (gitignored).
"""

from __future__ import annotations

import argparse
from pathlib import Path

# All Shakespeare character names to replace.
# Sorted alphabetically so Bob1 always maps to the same character.
SHAKESPEARE_NAMES: list[str] = sorted([
    "ADRIANA", "AEGEON", "AEMILIA", "AGAMEMNON", "AGRIPPA", "AJAX", "ALCIBIADES",
    "ALONSO", "ANGELO", "ANTONIO", "ARIEL", "ARMADO", "AUFIDIUS",
    "BALTHASAR", "BANQUO", "BAPTISTA", "BASSANIO", "BEATRICE", "BENEDICK",
    "BENVOLIO", "BIANCA", "BIONDELLO", "BOLINGBROKE", "BOTTOM", "BRUTUS",
    "CALIBAN", "CALPHURNIA", "CAPULET", "CASCA", "CASSIO", "CASSIUS",
    "CELIA", "CERIMON", "CESARIO", "CLAUDIUS", "CLEOPATRA", "CORDELIA",
    "CORNELIUS", "COSTARD", "CRESSIDA", "CYMBELINE",
    "DEMETRIUS", "DESDEMONA", "DIANA", "DOGBERRY", "DONALBAIN", "DUKE",
    "DUNCAN", "EDGAR", "EDMUND", "EMILIA", "ENOBARBUS",
    "FALSTAFF", "FESTE", "FLUTE", "FORD", "FORTINBRAS", "FRIAR",
    "GERTRUDE", "GONZALO", "GONERIL", "GRATIANO", "GREMIO",
    "HAMLET", "HECATE", "HELENA", "HENRY", "HERMIA", "HERMIONE",
    "HIPPOLYTA", "HORATIO", "HOTSPUR", "HYMEN",
    "IAGO", "IMOGEN", "IRIS",
    "JACQUENETTA", "JAQUES", "JESSICA", "JULIA", "JULIET", "JULIUS",
    "KATE", "KENT",
    "LAERTES", "LAUNCE", "LAUNCELOT", "LEAR", "LENNOX", "LEONATO",
    "LEONTES", "LEPIDUS", "LODOVICO", "LONGAVILLE", "LUCENTIO", "LUCIUS",
    "LYSANDER",
    "MACBETH", "MACDUFF", "MALCOLM", "MALVOLIO", "MARIANA", "MARINA",
    "MERCUTIO", "MIRANDA", "MONTAGUE", "MOTH",
    "NATHANIEL", "NERISSA", "OBERON", "OCTAVIA", "OCTAVIUS",
    "OLIVIA", "OPHELIA", "ORLANDO", "ORSINO", "OTHELLO",
    "PAGE", "PARIS", "PAULINA", "PERDITA", "PERICLES", "PETRUCHIO",
    "PHILOSTRATE", "PHRYNIA", "PISTOL", "POLONIUS", "PORTIA",
    "PROSPERO", "PROTEUS", "PUCK",
    "QUICKLY", "QUINCE",
    "REGAN", "RICHARD", "ROMEO", "ROSALIND", "ROSALINE", "ROSS",
    "SEBASTIAN", "SHYLOCK", "SILVIA", "SIMONIDES", "SNOUT", "SNUG",
    "STEPHANO", "STARVELING",
    "TAMING", "TIMON", "TITUS", "TITANIA", "TRANIO", "TRINCULO",
    "TROILUS", "TYBALT",
    "ULYSSES", "VALENTINE", "VINCENTIO", "VIOLA", "VIRGILIA",
    "WILLIAM",
])


def build_name_mapping(names: list[str]) -> dict[str, str]:
    """Map each name to BobN where N is its 1-based position in sorted order.

    Args:
        names: Already-sorted list of character names.

    Returns:
        Dict mapping each name to its Bob alias.
    """
    return {name: f"Bob{i + 1}" for i, name in enumerate(names)}


def bobify(text: str, mapping: dict[str, str]) -> str:
    """Replace all character name occurrences in text.

    Args:
        text: Raw input text.
        mapping: Dict of name → Bob alias.

    Returns:
        Text with all names replaced.
    """
    for name, alias in mapping.items():
        text = text.replace(name, alias)
    return text


def main() -> None:
    """Run the bobify preprocessing script."""
    parser = argparse.ArgumentParser(description="Replace Shakespeare names with Bob aliases.")
    parser.add_argument(
        "--input",
        default="data/raw/shakespeare.txt",
        help="Path to raw Shakespeare text (default: data/raw/shakespeare.txt)",
    )
    parser.add_argument(
        "--output",
        default="data/input.txt",
        help="Path to write bobified text (default: data/input.txt)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Download with:\n"
            "  mkdir -p data/raw\n"
            "  curl -o data/raw/shakespeare.txt "
            "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"
        )

    text = input_path.read_text(encoding="utf-8", errors="replace")
    mapping = build_name_mapping(SHAKESPEARE_NAMES)
    output = bobify(text, mapping)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")

    vocab_size = len(set(output))
    print(f"Wrote {len(output):,} chars to {output_path}")
    print(f"Vocab size: {vocab_size} unique characters")
    print(f"Update nano.yaml: vocab_size: {vocab_size}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Download raw Shakespeare and run bobify**

```bash
mkdir -p data/raw
curl -o data/raw/shakespeare.txt https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt
uv run python scripts/bobify.py
```

Expected output: something like
```
Wrote 5,458,199 chars to data/input.txt
Vocab size: 65 unique characters
Update nano.yaml: vocab_size: 65
```

- [ ] **Step 3: Update nano.yaml vocab_size to match Shakespeare vocab**

Edit `configs/nano.yaml` — replace `vocab_size: 26` with the value printed above (e.g. `vocab_size: 65`). Also update `max_seq_len` to a longer context window for Shakespeare training:

```yaml
model:
  vocab_size: 65        # ← update to match output of bobify.py
  d_model: 48
  n_heads: 3
  n_layers: 3
  d_ff: 192
  max_seq_len: 256      # ← longer context for Shakespeare
  rope_theta: 10000.0
  norm_eps: 1.0e-6
  tie_embeddings: true

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

- [ ] **Step 4: Commit**

```bash
git add scripts/bobify.py configs/nano.yaml
git commit -m "feat: add bobify preprocessing script, update nano.yaml for Shakespeare"
```

---

## Task 7: Train Entry Point

**Files:**
- Create: `scripts/train.py`

- [ ] **Step 1: Create train.py**

Create `scripts/train.py`:

```python
"""Entry point for training Bob on bobified Shakespeare text.

Usage:
    uv run python scripts/train.py --config configs/nano.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bob.config import ModelConfig, TrainingConfig
from bob.training.trainer import train


def main() -> None:
    """Parse args, detect device, and run the training loop."""
    parser = argparse.ArgumentParser(description="Train Bob on bobified Shakespeare.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    model_config = ModelConfig.from_yaml(config_path)
    training_config = TrainingConfig.from_yaml(config_path)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    train(model_config, training_config, device)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script runs end-to-end**

```bash
uv run python scripts/train.py --config configs/nano.yaml
```

Expected: prints `Using device: mps` (or `cpu`), then training logs every 500 steps:
```
step   500 | train_loss 2.1234 | val_loss 2.2345 | lr 3.00e-04 | 'Bob3\nTo be...'
```

- [ ] **Step 3: Run full test suite one final time**

```bash
uv run pytest -v
```

Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py
git commit -m "feat: add train.py entry point"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `scripts/bobify.py` with deterministic name→Bob mapping | Task 6 |
| `TrainingConfig` frozen dataclass + `from_yaml` | Task 1 |
| `configs/nano.yaml` extended with `training:` section | Task 1 + 6 |
| `CharDataset` with `(x, y)` shifted pairs | Task 2 |
| `build_dataloaders` returning `(train_loader, val_loader, tokenizer)` | Task 2 |
| `Tokenizer.chars` for vocab persistence | Task 2 |
| `get_lr` cosine warmup schedule | Task 3 |
| `save_checkpoint` / `load_latest_checkpoint` | Task 4 |
| `save_vocab` / `load_vocab` | Task 4 |
| `train()` with two param groups (decay/no-decay) | Task 5 |
| `train()` resumes from latest checkpoint | Task 5 |
| Eval every `eval_interval` steps with val loss + sample | Task 5 |
| Checkpoint every `checkpoint_interval` steps | Task 5 |
| `assert vocab_size == tokenizer.vocab_size` | Task 5 |
| `scripts/train.py` with `--config`, device detection | Task 7 |

**Placeholder scan:** None found.

**Type consistency:**
- `build_dataloaders` returns `(DataLoader, DataLoader, Tokenizer)` — consumed identically in `trainer.py`
- `save_vocab(tokenizer.chars, ...)` — `chars` property defined in Task 2, used in Task 5
- `load_latest_checkpoint(model, optimizer, config.checkpoint_dir)` — signature matches Task 4 definition
- `get_lr(step, config.warmup_steps, config.max_steps, config.learning_rate, config.min_lr)` — matches Task 3 definition
