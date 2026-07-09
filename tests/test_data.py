from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from src.data.partitioner import partition_dataset, partition_iid, partition_noniid_dirichlet


def _make_toy_dataset(num_samples: int = 100) -> TensorDataset:
    x = torch.randn(num_samples, 1, 8, 8)
    y = torch.randint(0, 10, (num_samples,))
    return TensorDataset(x, y)


def test_partition_iid_equal_split() -> None:
    dataset = _make_toy_dataset(100)
    subsets = partition_iid(dataset, 5)
    sizes = [len(s) for s in subsets]
    assert len(subsets) == 5
    assert sum(sizes) == 100


def test_partition_noniid_returns_correct_count() -> None:
    dataset = _make_toy_dataset(200)
    subsets = partition_noniid_dirichlet(dataset, 5, alpha=0.5)
    assert len(subsets) == 5
    total = sum(len(s) for s in subsets)
    assert total == 200


def test_partition_noniid_no_empty_clients() -> None:
    x = torch.randn(50, 1, 8, 8)
    y = torch.tensor([0] + [1] * 49)
    dataset = TensorDataset(x, y)
    subsets = partition_noniid_dirichlet(dataset, 50, alpha=0.1)
    assert len(subsets) == 50
    assert all(len(s) >= 1 for s in subsets)
    assert sum(len(s) for s in subsets) == 50


def test_partition_dataset_factory_iid() -> None:
    dataset = _make_toy_dataset(50)
    subsets = partition_dataset(dataset, 5, partition_type="iid")
    assert len(subsets) == 5


def test_partition_dataset_factory_noniid() -> None:
    dataset = _make_toy_dataset(50)
    subsets = partition_dataset(dataset, 5, partition_type="noniid", alpha=1.0)
    assert len(subsets) == 5
