"""Tests for the KV cache."""

import pytest
import torch

from bob.config import ModelConfig
from bob.inference.kv_cache import KVCache


def _config(n_layers: int = 2, max_seq_len: int = 16) -> ModelConfig:
    # kv_heads=2, d_head=8
    return ModelConfig(
        vocab_size=50,
        d_model=16,
        n_heads=4,
        n_layers=n_layers,
        d_ff=64,
        max_seq_len=max_seq_len,
        n_kv_heads=2,
        head_dim=8,
    )


def _cache(n_layers: int = 2, max_seq_len: int = 16) -> KVCache:
    return KVCache(
        _config(n_layers, max_seq_len),
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_preallocates_full_buffer() -> None:
    cache = _cache(n_layers=2, max_seq_len=16)
    assert cache.k_cache[0].shape == (1, 2, 16, 8)  # (B, n_kv_heads, max_seq_len, d_head)


def test_update_writes_at_cache_position() -> None:
    cache = _cache(n_layers=1, max_seq_len=16)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    k_all, v_all = cache.update(0, k, v, past_seen_tokens=0)
    assert k_all.shape == (1, 2, 4, 8)
    torch.testing.assert_close(k_all, k)
    torch.testing.assert_close(v_all, v)


def test_second_update_appends_at_next_position() -> None:
    cache = _cache(n_layers=1, max_seq_len=16)
    k1 = torch.randn(1, 2, 4, 8)
    v1 = torch.randn(1, 2, 4, 8)
    cache.update(0, k1, v1, past_seen_tokens=0)
    k2 = torch.randn(1, 2, 1, 8)
    v2 = torch.randn(1, 2, 1, 8)
    k_all, _ = cache.update(0, k2, v2, past_seen_tokens=4)
    assert k_all.shape == (1, 2, 5, 8)
    torch.testing.assert_close(k_all[:, :, :4], k1)
    torch.testing.assert_close(k_all[:, :, 4:], k2)


def test_layers_share_position_not_data() -> None:
    cache = _cache(n_layers=2, max_seq_len=16)
    k0 = torch.randn(1, 2, 3, 8)
    cache.update(0, k0, k0, past_seen_tokens=0)
    k1 = torch.randn(1, 2, 3, 8)
    a, _ = cache.update(1, k1, k1, past_seen_tokens=0)  # same window, layer 1's own data
    assert a.shape == (1, 2, 3, 8)
    torch.testing.assert_close(a, k1)


def test_overflow_raises() -> None:
    cache = _cache(n_layers=1, max_seq_len=4)
    k = torch.randn(1, 2, 3, 8)
    cache.update(0, k, k, past_seen_tokens=0)
    with pytest.raises(ValueError):
        cache.update(0, k, k, past_seen_tokens=3)  # 3 + 3 > 4
