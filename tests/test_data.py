from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from src.data.partitioner import (
    partition_dataset,
    partition_iid,
    partition_noniid_dirichlet,
    partition_single,
)


def _make_toy_dataset(num_samples: int = 100) -> TensorDataset:
    x = torch.randn(num_samples, 1, 8, 8)
    y = torch.randint(0, 10, (num_samples,))
    return TensorDataset(x, y)


def _make_toy_dataset_with_classes(
    num_samples: int, num_classes: int,
) -> TensorDataset:
    x = torch.randn(num_samples, 1, 8, 8)
    y = torch.randint(0, num_classes, (num_samples,))
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


def test_partition_noniid_fewer_classes_than_clients() -> None:
    dataset = _make_toy_dataset_with_classes(200, 3)
    subsets = partition_noniid_dirichlet(dataset, 10, alpha=1.0)
    assert len(subsets) == 10
    assert all(len(s) >= 1 for s in subsets)
    assert sum(len(s) for s in subsets) == 200


def test_partition_noniid_single_class_dataset() -> None:
    dataset = _make_toy_dataset_with_classes(100, 1)
    subsets = partition_noniid_dirichlet(dataset, 5, alpha=1.0)
    assert len(subsets) == 5
    assert all(len(s) >= 1 for s in subsets)
    assert sum(len(s) for s in subsets) == 100


def test_partition_noniid_extreme_alpha() -> None:
    dataset = _make_toy_dataset_with_classes(200, 5)
    subsets = partition_noniid_dirichlet(dataset, 50, alpha=0.001)
    assert len(subsets) == 50
    assert all(len(s) >= 1 for s in subsets)
    assert sum(len(s) for s in subsets) == 200


def test_partition_noniid_large_alpha() -> None:
    dataset = _make_toy_dataset_with_classes(1000, 5)
    subsets = partition_noniid_dirichlet(dataset, 5, alpha=100.0)
    assert len(subsets) == 5
    assert all(len(s) >= 1 for s in subsets)
    assert sum(len(s) for s in subsets) == 1000
    sizes = [len(s) for s in subsets]
    assert max(sizes) - min(sizes) < 0.3 * sum(sizes) / len(sizes)


def test_partition_noniid_dirichlet_seed_reproducibility() -> None:
    dataset = _make_toy_dataset(200)
    a = partition_noniid_dirichlet(dataset, 5, alpha=0.5, seed=42)
    b = partition_noniid_dirichlet(dataset, 5, alpha=0.5, seed=42)
    for sa, sb in zip(a, b, strict=True):
        assert sa.indices == sb.indices


def test_partition_noniid_dirichlet_different_seed_different() -> None:
    dataset = _make_toy_dataset(200)
    a = partition_noniid_dirichlet(dataset, 5, alpha=0.5, seed=42)
    b = partition_noniid_dirichlet(dataset, 5, alpha=0.5, seed=99)
    assert any(sa.indices != sb.indices for sa, sb in zip(a, b, strict=True))


class TestPartitionSingle:
    def test_iid_returns_correct_size(self) -> None:
        dataset = _make_toy_dataset(100)
        subset = partition_single(dataset, 5, 0, partition_type="iid", seed=42)
        assert len(subset) == pytest.approx(20, abs=5)

    def test_noniid_returns_correct_size(self) -> None:
        dataset = _make_toy_dataset(200)
        subset = partition_single(dataset, 5, 0, partition_type="noniid", seed=42)
        assert len(subset) > 0

    def test_raises_for_negative_id(self) -> None:
        dataset = _make_toy_dataset(50)
        with pytest.raises(ValueError, match="out of range"):
            partition_single(dataset, 5, -1)

    def test_raises_for_overflow_id(self) -> None:
        dataset = _make_toy_dataset(50)
        with pytest.raises(ValueError, match="out of range"):
            partition_single(dataset, 5, 5)

    def test_consistency_with_same_seed(self) -> None:
        dataset = _make_toy_dataset(100)
        a = partition_single(dataset, 5, 0, partition_type="iid", seed=42)
        b = partition_single(dataset, 5, 0, partition_type="iid", seed=42)
        assert a.indices == b.indices

    def test_different_seeds_different(self) -> None:
        dataset = _make_toy_dataset(100)
        a = partition_single(dataset, 5, 0, partition_type="iid", seed=42)
        b = partition_single(dataset, 5, 0, partition_type="iid", seed=99)
        assert a.indices != b.indices

    def test_iid_partitions_are_disjoint(self) -> None:
        dataset = _make_toy_dataset(100)
        parts = [
            set(partition_single(dataset, 5, i, partition_type="iid", seed=42).indices)
            for i in range(5)
        ]
        for i in range(5):
            for j in range(i + 1, 5):
                assert parts[i].isdisjoint(parts[j])

    def test_noniid_returns_valid_subset(self) -> None:
        dataset = _make_toy_dataset(200)
        subset = partition_single(dataset, 5, 0, partition_type="noniid", seed=42)
        assert len(subset) > 0
        assert len(subset) < 200
        assert all(0 <= idx < 200 for idx in subset.indices)

    def test_noniid_extreme_alpha_nonempty(self) -> None:
        dataset = _make_toy_dataset_with_classes(50, 10)
        subset = partition_single(dataset, 50, 0, partition_type="noniid", alpha=0.001, seed=42)
        assert len(subset) >= 1
