"""Tests for cosine LR schedule with linear warmup."""

import math

import pytest

from bob.training.schedule import get_lr


def test_get_lr_at_step_zero() -> None:
    lr = get_lr(step=0, warmup_steps=100, max_steps=1000, max_lr=3e-4, min_lr=3e-5)
    assert lr == 0.0


def test_get_lr_warmup_midpoint() -> None:
    lr = get_lr(step=50, warmup_steps=100, max_steps=1000, max_lr=3e-4, min_lr=3e-5)
    assert abs(lr - 1.5e-4) < 1e-12


def test_get_lr_end_of_warmup() -> None:
    lr = get_lr(step=100, warmup_steps=100, max_steps=1000, max_lr=3e-4, min_lr=3e-5)
    assert abs(lr - 3e-4) < 1e-12


def test_get_lr_cosine_midpoint() -> None:
    # progress = (550 - 100) / (1000 - 100) = 0.5
    # cos(pi * 0.5) = 0 → lr = min_lr + 0.5*(max_lr - min_lr)*(1+0)
    lr = get_lr(step=550, warmup_steps=100, max_steps=1000, max_lr=3e-4, min_lr=3e-5)
    expected = 3e-5 + 0.5 * (3e-4 - 3e-5) * (1.0 + math.cos(math.pi * 0.5))
    assert abs(lr - expected) < 1e-12


def test_get_lr_at_max_steps_raises() -> None:
    with pytest.raises(AssertionError):
        get_lr(step=1000, warmup_steps=100, max_steps=1000, max_lr=3e-4, min_lr=3e-5)


def test_get_lr_beyond_max_steps_raises() -> None:
    with pytest.raises(AssertionError):
        get_lr(step=9999, warmup_steps=100, max_steps=1000, max_lr=3e-4, min_lr=3e-5)


def test_get_lr_is_monotone_decreasing_after_warmup() -> None:
    steps = list(range(100, 1000, 50))  # up to 999 (last valid step when max_steps=1000)
    lrs = [get_lr(s, warmup_steps=100, max_steps=1000, max_lr=3e-4, min_lr=3e-5) for s in steps]
    for a, b in zip(lrs, lrs[1:], strict=False):
        assert a >= b
