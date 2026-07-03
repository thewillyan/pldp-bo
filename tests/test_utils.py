from __future__ import annotations

import random

import numpy as np
import torch

from src.utils import set_seed


def test_set_seed_torch_reproducible() -> None:
    set_seed(42)
    a1 = torch.randn(5)
    set_seed(42)
    a2 = torch.randn(5)
    assert torch.allclose(a1, a2)


def test_set_seed_numpy_reproducible() -> None:
    set_seed(42)
    a1 = np.random.randn(5)
    set_seed(42)
    a2 = np.random.randn(5)
    assert np.allclose(a1, a2)


def test_set_seed_python_random_reproducible() -> None:
    set_seed(42)
    a1 = [random.random() for _ in range(5)]
    set_seed(42)
    a2 = [random.random() for _ in range(5)]
    assert a1 == a2


def test_set_seed_different_seeds_differ() -> None:
    set_seed(42)
    a = torch.randn(5)
    set_seed(99)
    b = torch.randn(5)
    assert not torch.allclose(a, b)


def test_set_seed_deterministic_flag() -> None:
    set_seed(42, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    set_seed(42, deterministic=False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
