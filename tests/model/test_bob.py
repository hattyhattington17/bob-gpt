"""End-to-end forward/backward checks for the Phase A architecture."""

from pathlib import Path

import torch

from bob.config import ModelConfig
from bob.model.transformer import Bob


def test_nano_config_forwards() -> None:
    config = ModelConfig.from_yaml(Path("configs/nano.yaml"))
    model = Bob(config)
    seq_len = 12
    ids = torch.randint(0, config.vocab_size, (2, seq_len))
    logits = model(ids, padding_mask=torch.ones(2, seq_len, dtype=torch.long))
    assert logits.shape == (2, seq_len, config.vocab_size)


def test_gqa_qk_norm_config_forwards() -> None:
    config = ModelConfig(
        vocab_size=50,
        d_model=32,
        n_heads=8,
        n_layers=2,
        d_ff=64,
        max_seq_len=16,
        n_kv_heads=2,
        head_dim=4,
        qk_norm=True,
    )
    model = Bob(config)
    seq_len = 10
    ids = torch.randint(0, config.vocab_size, (3, seq_len))
    logits = model(ids, padding_mask=torch.ones(3, seq_len, dtype=torch.long))
    assert logits.shape == (3, seq_len, config.vocab_size)


def test_backward_produces_finite_grads() -> None:
    config = ModelConfig(
        vocab_size=50,
        d_model=32,
        n_heads=8,
        n_layers=2,
        d_ff=64,
        max_seq_len=16,
        n_kv_heads=2,
        head_dim=4,
        qk_norm=True,
    )
    model = Bob(config)
    ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(ids, padding_mask=torch.ones(2, 8, dtype=torch.long))
    loss = logits.float().log_softmax(dim=-1).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients populated"
    assert all(torch.isfinite(g).all() for g in grads)
