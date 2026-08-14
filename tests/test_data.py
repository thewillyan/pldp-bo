from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch
from torch.utils.data import TensorDataset

from src.config.loader import DataConfig
from src.data.partitioner import (
    build_partition_kwargs,
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


def _make_balanced_dataset(num_samples: int, num_classes: int) -> TensorDataset:
    x = torch.randn(num_samples, 1, 8, 8)
    y = torch.arange(num_samples) % num_classes
    return TensorDataset(x, y)


def _client_class_counts(indices: Sequence[int], dataset: TensorDataset) -> dict[int, int]:
    targets = dataset.tensors[1]
    counts: dict[int, int] = {}
    for idx in indices:
        c = int(targets[idx])
        counts[c] = counts.get(c, 0) + 1
    return counts


class TestPartitionTypes:
    def test_noniid_alias_matches_dirichlet_alpha05(self) -> None:
        dataset = _make_toy_dataset(200)
        alias = partition_dataset(dataset, 5, "noniid", seed=42)
        explicit = partition_dataset(dataset, 5, "dirichlet", alpha=0.5, seed=42)
        assert [s.indices for s in alias] == [s.indices for s in explicit]

    def test_dirichlet_accepted(self) -> None:
        dataset = _make_toy_dataset(200)
        subsets = partition_dataset(dataset, 5, "dirichlet", alpha=0.1, seed=42)
        assert len(subsets) == 5
        assert sum(len(s) for s in subsets) == 200

    def test_writer_raises_not_implemented(self) -> None:
        dataset = _make_toy_dataset(100)
        with pytest.raises(NotImplementedError, match="IMPL-07"):
            partition_dataset(dataset, 5, "writer", seed=42)

    def test_unknown_type_raises(self) -> None:
        dataset = _make_toy_dataset(100)
        with pytest.raises(ValueError, match="Unknown partition type"):
            partition_dataset(dataset, 5, "unknown", seed=42)


class TestPathological:
    def test_two_classes_per_client_balanced(self) -> None:
        dataset = _make_balanced_dataset(12000, 10)
        subsets = partition_dataset(dataset, 100, "pathological", seed=42)
        assert len(subsets) == 100
        all_idx: set[int] = set()
        for s in subsets:
            counts = _client_class_counts(s.indices, dataset)
            assert len(counts) == 2
            assert set(counts.values()) == {60}  # 1200/class ÷ 20 covering clients
            all_idx |= set(s.indices)
        assert len(all_idx) == 12000  # every sample assigned exactly once

    def test_class_coverage_mnist_cell(self) -> None:
        dataset = _make_balanced_dataset(12000, 10)
        subsets = partition_dataset(dataset, 100, "pathological", seed=42)
        cover = [0] * 10
        for s in subsets:
            for c in _client_class_counts(s.indices, dataset):
                cover[c] += 1
        assert cover == [20] * 10  # 2K/C = 20 clients per class

    def test_class_coverage_cifar100_cell(self) -> None:
        dataset = _make_balanced_dataset(10000, 100)
        subsets = partition_dataset(dataset, 100, "pathological", seed=42)
        cover = [0] * 100
        for s in subsets:
            for c in _client_class_counts(s.indices, dataset):
                cover[c] += 1
        assert cover == [2] * 100  # 2K/C = 2 clients per class

    def test_deterministic(self) -> None:
        dataset = _make_balanced_dataset(12000, 10)
        a = partition_dataset(dataset, 100, "pathological", seed=42)
        b = partition_dataset(dataset, 100, "pathological", seed=42)
        c = partition_dataset(dataset, 100, "pathological", seed=7)
        assert [s.indices for s in a] == [s.indices for s in b]
        assert [s.indices for s in a] != [s.indices for s in c]


class TestMinSamples:
    def test_dirichlet_tops_up_deficient_clients(self) -> None:
        dataset = _make_toy_dataset_with_classes(1000, 10)
        subsets = partition_dataset(
            dataset, 20, "dirichlet", alpha=0.001, seed=42, min_samples=30,
        )
        sizes = [len(s) for s in subsets]
        assert len(subsets) == 20
        assert all(s >= 30 for s in sizes)
        assert sum(sizes) == 1000

    def test_iid_tops_up_deficient_clients(self) -> None:
        dataset = _make_toy_dataset(1000)
        subsets = partition_dataset(dataset, 20, "iid", seed=42, min_samples=30)
        sizes = [len(s) for s in subsets]
        assert all(s >= 30 for s in sizes)
        assert sum(sizes) == 1000

    def test_deterministic(self) -> None:
        dataset = _make_toy_dataset_with_classes(1000, 10)
        a = partition_dataset(
            dataset, 20, "dirichlet", alpha=0.001, seed=42, min_samples=30,
        )
        b = partition_dataset(
            dataset, 20, "dirichlet", alpha=0.001, seed=42, min_samples=30,
        )
        assert [s.indices for s in a] == [s.indices for s in b]

    def test_disabled_when_zero(self) -> None:
        dataset = _make_toy_dataset(200)
        subsets = partition_dataset(dataset, 5, "iid", seed=42, min_samples=0)
        assert sum(len(s) for s in subsets) == 200


class TestSingleFullParity:
    def test_single_matches_full_for_all_types(self) -> None:
        for ptype, alpha in (("iid", 1.0), ("dirichlet", 0.1), ("pathological", 1.0)):
            dataset = _make_balanced_dataset(1000, 10)
            full = partition_dataset(
                dataset, 20, ptype, alpha=alpha, seed=42, min_samples=30,
            )
            for i in range(20):
                single = partition_single(
                    dataset, 20, i, ptype, alpha=alpha, seed=42, min_samples=30,
                )
                assert single.indices == full[i].indices, (ptype, i)


class TestPartitionKwargs:
    def test_build_partition_kwargs(self) -> None:
        assert build_partition_kwargs("iid") == {"type": "iid"}
        assert build_partition_kwargs("dirichlet", alpha=0.5) == {
            "type": "dirichlet", "alpha": 0.5,
        }
        assert build_partition_kwargs("noniid") == {
            "type": "dirichlet", "alpha": 0.5,
        }
        assert build_partition_kwargs("pathological") == {
            "type": "pathological", "classes_per_client": 2,
        }


def test_partition_min_samples_default() -> None:
    from src.config.loader import ExperimentConfig
    assert ExperimentConfig().data.partition_min_samples == 30


class TestSplitHoldout:
    def test_splits_by_frac(self) -> None:
        from torch.utils.data import Subset

        from src.data.partitioner import split_holdout
        dataset = _make_toy_dataset(100)
        subset = Subset(dataset, list(range(100)))
        train, val = split_holdout(subset, 0.1, seed=7)
        assert len(val) == 10
        assert len(train) == 90

    def test_disjoint_and_exhaustive(self) -> None:
        from torch.utils.data import Subset

        from src.data.partitioner import split_holdout
        dataset = _make_toy_dataset(100)
        subset = Subset(dataset, list(range(100)))
        train, val = split_holdout(subset, 0.1, seed=7)
        assert set(train.indices).isdisjoint(val.indices)
        assert sorted(set(train.indices) | set(val.indices)) == list(range(100))

    def test_deterministic_given_seed(self) -> None:
        from torch.utils.data import Subset

        from src.data.partitioner import split_holdout
        dataset = _make_toy_dataset(100)
        subset = Subset(dataset, list(range(100)))
        train_a, val_a = split_holdout(subset, 0.1, seed=7)
        train_b, val_b = split_holdout(subset, 0.1, seed=7)
        assert train_a.indices == train_b.indices
        assert val_a.indices == val_b.indices

    def test_different_seed_different_holdout(self) -> None:
        from torch.utils.data import Subset

        from src.data.partitioner import split_holdout
        dataset = _make_toy_dataset(100)
        subset = Subset(dataset, list(range(100)))
        _, val_a = split_holdout(subset, 0.1, seed=7)
        _, val_b = split_holdout(subset, 0.1, seed=8)
        assert val_a.indices != val_b.indices

    def test_rejects_bad_frac(self) -> None:
        from torch.utils.data import Subset

        from src.data.partitioner import split_holdout
        dataset = _make_toy_dataset(100)
        subset = Subset(dataset, list(range(100)))
        with pytest.raises(ValueError, match="val_frac"):
            split_holdout(subset, 1.0, seed=7)
        with pytest.raises(ValueError, match="val_frac"):
            split_holdout(subset, -0.1, seed=7)

    def test_zero_frac_no_holdout(self) -> None:
        from torch.utils.data import Subset

        from src.data.partitioner import split_holdout
        dataset = _make_toy_dataset(100)
        subset = Subset(dataset, list(range(100)))
        train, val = split_holdout(subset, 0.0, seed=7)
        assert len(val) == 0
        assert sorted(train.indices) == list(range(100))


def _make_config_for_loader(**kwargs: object) -> "DataConfig":
    from src.config.loader import DataConfig
    cfg = DataConfig(name="mnist", data_dir="/tmp/opencode/fake-data")
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


class TestCreateClientDataloader:
    """Hold-out + loader plumbing of create_client_dataloader (IMPL-06)."""

    def _make(self, monkeypatch: pytest.MonkeyPatch, num_samples: int = 200,
              num_clients: int = 4, partition_id: int = 0, seed: int = 42) -> tuple[Any, ...]:
        from src.data import create_client_dataloader
        dataset = _make_toy_dataset(num_samples)
        monkeypatch.setattr("src.data._cached_dataset", lambda _name, _data_dir: dataset)
        cfg = _make_config_for_loader(num_clients=num_clients, batch_size=8)
        return create_client_dataloader(cfg, partition_id, num_clients, seed=seed)

    def test_returns_five_tuple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._make(monkeypatch)
        assert len(result) == 5
        train_loader, val_loader, train_subset, val_subset, total = result
        assert total == 200
        assert len(train_loader.dataset) == len(train_subset)
        assert len(val_loader.dataset) == len(val_subset)

    def test_train_subset_is_holdout_fraction_of_partition(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, _, train_subset, val_subset, _ = self._make(monkeypatch)
        assert len(train_subset) == 45  # 50-sample iid partition, 10% held out
        assert len(val_subset) == 5

    def test_train_val_indices_disjoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, _, train_subset, val_subset, _ = self._make(monkeypatch)
        assert set(train_subset.indices).isdisjoint(val_subset.indices)

    def test_holdout_fixed_across_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.data import create_client_dataloader
        dataset = _make_toy_dataset(200)
        monkeypatch.setattr("src.data._cached_dataset", lambda _name, _data_dir: dataset)
        cfg = _make_config_for_loader(num_clients=4, batch_size=8)
        first = create_client_dataloader(cfg, 0, 4, seed=42)
        second = create_client_dataloader(cfg, 0, 4, seed=42)
        assert first[2].indices == second[2].indices
        assert first[3].indices == second[3].indices

    def test_holdout_differs_per_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.data import create_client_dataloader
        dataset = _make_toy_dataset(200)
        monkeypatch.setattr("src.data._cached_dataset", lambda _name, _data_dir: dataset)
        cfg = _make_config_for_loader(num_clients=4, batch_size=8)
        client_a = create_client_dataloader(cfg, 0, 4, seed=42)
        client_b = create_client_dataloader(cfg, 1, 4, seed=42)
        assert set(client_a[2].indices).isdisjoint(client_b[2].indices)
        assert client_a[3].indices != client_b[3].indices

    def test_holdout_never_trained_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        train_loader, _, train_subset, val_subset, _ = self._make(monkeypatch)
        assert train_loader.dataset is train_subset
        assert set(train_subset.indices).isdisjoint(val_subset.indices)

    def test_total_train_size_is_global(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dataset = _make_toy_dataset(200)
        monkeypatch.setattr("src.data._cached_dataset", lambda _name, _data_dir: dataset)
        cfg = _make_config_for_loader(num_clients=4, batch_size=8)
        from src.data import create_client_dataloader
        _, _, _, _, total = create_client_dataloader(cfg, 2, 4, seed=42)
        assert total == 200


def test_create_dataset_returns_full_train_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import create_dataset

    class FakeDataset:
        def __init__(self, root: str, train: bool, transform: object, download: bool) -> None:
            self._root, self._train, self._transform, self._download = (
                root, train, transform, download,
            )
            self._data = torch.randn(40, 1, 8, 8)
            self._labels = torch.randint(0, 10, (40,))

        def __len__(self) -> int:
            return 40

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self._data[idx], self._labels[idx]

    monkeypatch.setattr("src.data.DATASET_REGISTRY", {"mnist": FakeDataset})
    cfg = _make_config_for_loader()
    dataset = create_dataset(cfg)
    assert len(dataset) == 40  # type: ignore[arg-type]
