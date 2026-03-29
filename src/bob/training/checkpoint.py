"""Save and load model checkpoints and vocabulary."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def save_best_checkpoint(
    validation_loss: float,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str | Path,
) -> None:
    """Save model and optimizer state to best.pt, overwriting any previous best.

    Args:
        validation_loss: Validation loss achieved at this checkpoint.
        step: Current training step.
        model: Model whose state_dict to save.
        optimizer: Optimizer whose state_dict to save.
        checkpoint_dir: Directory to write the checkpoint to.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir) / "best.pt"
    torch.save(
        {
            "validation_loss": validation_loss,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_best_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str | Path,
) -> tuple[int, float]:
    """Load best.pt if it exists.

    Args:
        model: Model to load state into.
        optimizer: Optimizer to load state into.
        checkpoint_dir: Directory to search for best.pt.

    Returns:
        Tuple of (step, best_validation_loss). Returns (0, inf) if no checkpoint found.
    """
    ckpt_dir = Path(checkpoint_dir)
    path = ckpt_dir / "best.pt"
    if not path.exists():
        return 0, math.inf
    # weights_only=False required for unpickler to handle saved progress file
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt["step"]), float(ckpt["validation_loss"])


def save_vocab(chars: list[str], checkpoint_dir: str | Path) -> None:
    """Persist the tokenizer character list to vocab.json.

    Args:
        chars: List of characters from the tokenizer, in ID order.
        checkpoint_dir: Directory to write vocab.json to.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir) / "vocab.json"
    path.write_text(json.dumps(chars))


def load_vocab(checkpoint_dir: str | Path) -> list[str]:
    """Load the character list from vocab.json.

    Args:
        checkpoint_dir: Directory containing vocab.json.

    Returns:
        List of characters in ID order.
    """
    path = Path(checkpoint_dir) / "vocab.json"
    return json.loads(path.read_text())  # type: ignore[no-any-return]
