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
    from max_lr to min_lr over the remaining steps. Raises if step >= max_steps.

    Args:
        step: Current training step (0-indexed).
        warmup_steps: Number of warmup steps.
        max_steps: Total training steps.
        max_lr: Peak learning rate.
        min_lr: Minimum learning rate (floor of cosine decay).

    Returns:
        Learning rate for this step.
    """
    assert step < max_steps, f"step {step} >= max_steps {max_steps}"
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
