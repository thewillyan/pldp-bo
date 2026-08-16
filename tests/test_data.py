from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from src.data.dataloaders import (
    DATASET_REGISTRY,
    NUM_CLASSES_MAP,
    TRANSFORMS_MAP,
    get_num_classes,
)
from src.data.femnist import FEMNISTDataset, femnist_counts
from src.data.partitioner import (
    build_partition_kwargs,
    partition_dataset,
    partition_iid,
    partition_noniid_dirichlet,
    partition_single,
)
from torch.utils.data import TensorDataset

from src.config.loader import DataConfig


def _make_toy_dataset(num_samples: int = 100) -> TensorDataset:
    x = torch.randn(num_samples, 1, 8, 8)
    y = torch.randint(0, 10, (num_samples,))
    return TensorDataset(x, y)


def _make_toy_dataset_with_classes(
    num_samples: int,
    num_classes: int,
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

    def test_writer_requires_users_attr(self) -> None:
        dataset = _make_toy_dataset(100)
        with pytest.raises(ValueError, match="users"):
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
            dataset,
            20,
            "dirichlet",
            alpha=0.001,
            seed=42,
            min_samples=30,
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
            dataset,
            20,
            "dirichlet",
            alpha=0.001,
            seed=42,
            min_samples=30,
        )
        b = partition_dataset(
            dataset,
            20,
            "dirichlet",
            alpha=0.001,
            seed=42,
            min_samples=30,
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
                dataset,
                20,
                ptype,
                alpha=alpha,
                seed=42,
                min_samples=30,
            )
            for i in range(20):
                single = partition_single(
                    dataset,
                    20,
                    i,
                    ptype,
                    alpha=alpha,
                    seed=42,
                    min_samples=30,
                )
                assert single.indices == full[i].indices, (ptype, i)


class TestPartitionKwargs:
    def test_build_partition_kwargs(self) -> None:
        assert build_partition_kwargs("iid") == {"type": "iid"}
        assert build_partition_kwargs("dirichlet", alpha=0.5) == {
            "type": "dirichlet",
            "alpha": 0.5,
        }
        assert build_partition_kwargs("noniid") == {
            "type": "dirichlet",
            "alpha": 0.5,
        }
        assert build_partition_kwargs("pathological") == {
            "type": "pathological",
            "classes_per_client": 2,
        }


def test_partition_min_samples_default() -> None:
    from src.config.loader import ExperimentConfig

    assert ExperimentConfig().data.partition_min_samples == 30


class TestSplitHoldout:
    def test_splits_by_frac(self) -> None:
        from src.data.partitioner import split_holdout
        from torch.utils.data import Subset

        dataset = _make_toy_dataset(100)
        subset = Subset(dataset, list(range(100)))
        train, val = split_holdout(subset, 0.1, seed=7)
        assert len(val) == 10
        assert len(train) == 90

    def test_disjoint_and_exhaustive(self) -> None:
        from src.data.partitioner import split_holdout
        from torch.utils.data import Subset

        dataset = _make_toy_dataset(100)
        subset = Subset(dataset, list(range(100)))
        train, val = split_holdout(subset, 0.1, seed=7)
        assert set(train.indices).isdisjoint(val.indices)
        assert sorted(set(train.indices) | set(val.indices)) == list(range(100))

    def test_deterministic_given_seed(self) -> None:
        from src.data.partitioner import split_holdout
        from torch.utils.data import Subset

        dataset = _make_toy_dataset(100)
        subset = Subset(dataset, list(range(100)))
        train_a, val_a = split_holdout(subset, 0.1, seed=7)
        train_b, val_b = split_holdout(subset, 0.1, seed=7)
        assert train_a.indices == train_b.indices
        assert val_a.indices == val_b.indices

    def test_different_seed_different_holdout(self) -> None:
        from src.data.partitioner import split_holdout
        from torch.utils.data import Subset

        dataset = _make_toy_dataset(100)
        subset = Subset(dataset, list(range(100)))
        _, val_a = split_holdout(subset, 0.1, seed=7)
        _, val_b = split_holdout(subset, 0.1, seed=8)
        assert val_a.indices != val_b.indices

    def test_rejects_bad_frac(self) -> None:
        from src.data.partitioner import split_holdout
        from torch.utils.data import Subset

        dataset = _make_toy_dataset(100)
        subset = Subset(dataset, list(range(100)))
        with pytest.raises(ValueError, match="val_frac"):
            split_holdout(subset, 1.0, seed=7)
        with pytest.raises(ValueError, match="val_frac"):
            split_holdout(subset, -0.1, seed=7)

    def test_zero_frac_no_holdout(self) -> None:
        from src.data.partitioner import split_holdout
        from torch.utils.data import Subset

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

    def _make(
        self,
        monkeypatch: pytest.MonkeyPatch,
        num_samples: int = 200,
        num_clients: int = 4,
        partition_id: int = 0,
        seed: int = 42,
    ) -> tuple[Any, ...]:
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
        self,
        monkeypatch: pytest.MonkeyPatch,
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
                root,
                train,
                transform,
                download,
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


class _RecordingRegistryDataset:
    """Registry stand-in that records the ``train`` flag of each construction."""

    seen: list[bool] = []

    def __init__(self, root: str, train: bool, transform: object, download: bool) -> None:
        _RecordingRegistryDataset.seen.append(train)
        self._root = root
        self._transform = transform
        self._download = download
        self._data = torch.randn(10, 1, 8, 8)
        self._labels = torch.randint(0, 10, (10,))

    def __len__(self) -> int:
        return 10

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._data[idx], self._labels[idx]


def test_create_dataset_respects_train_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.data import create_dataset

    _RecordingRegistryDataset.seen = []
    monkeypatch.setattr("src.data.DATASET_REGISTRY", {"mnist": _RecordingRegistryDataset})
    cfg = DataConfig(name="mnist", data_dir=str(tmp_path))
    create_dataset(cfg)
    create_dataset(cfg, train=False)
    assert _RecordingRegistryDataset.seen == [True, False]


def test_cached_dataset_keys_by_train_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.data import create_dataset, create_test_dataset

    _RecordingRegistryDataset.seen = []
    monkeypatch.setattr("src.data.DATASET_REGISTRY", {"mnist": _RecordingRegistryDataset})
    cfg = DataConfig(name="mnist", data_dir=str(tmp_path))
    create_dataset(cfg)
    ds_test = create_test_dataset(cfg)
    assert _RecordingRegistryDataset.seen == [True, False]
    assert create_test_dataset(cfg) is ds_test
    assert create_dataset(cfg) is not ds_test
    assert _RecordingRegistryDataset.seen == [True, False, True]


def test_create_test_loader_uses_official_test_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.data import create_test_loader

    _RecordingRegistryDataset.seen = []
    monkeypatch.setattr("src.data.DATASET_REGISTRY", {"mnist": _RecordingRegistryDataset})
    cfg = DataConfig(name="mnist", data_dir=str(tmp_path), batch_size=4)
    loader = create_test_loader(cfg)
    assert _RecordingRegistryDataset.seen == [False]
    assert loader.batch_size == 4
    from torch.utils.data import SequentialSampler

    assert isinstance(loader.sampler, SequentialSampler)  # no shuffle
    assert len(cast(Any, loader).dataset) == 10


def _make_writer_dataset(
    writer_sizes: Sequence[int],
    writer_labels: Sequence[Sequence[int]],
) -> TensorDataset:
    x_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    u_parts: list[torch.Tensor] = []
    for w, (n, labels) in enumerate(zip(writer_sizes, writer_labels, strict=True)):
        assert len(labels) == n
        x_parts.append(torch.randn(n, 1, 8, 8))
        y_parts.append(torch.tensor(labels))
        u_parts.append(torch.full((n,), w, dtype=torch.long))
    dataset = TensorDataset(torch.cat(x_parts), torch.cat(y_parts))
    cast(Any, dataset).users = torch.cat(u_parts)
    return dataset


def _users_of(dataset: TensorDataset) -> torch.Tensor:
    return cast(torch.Tensor, cast(Any, dataset).users)


def _write_fake_femnist(
    root: Path,
    n_train: int = 100,
    n_test: int = 20,
    n_writers: int = 5,
    scale: float = 1.0,
) -> None:
    processed = root / "FEMNIST" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    torch.save(
        [
            torch.full((n_train, 28, 28), 200.0 * scale),
            torch.randint(0, 62, (n_train,)),
            torch.randint(0, n_writers, (n_train,)),
        ],
        processed / "femnist_train.pt",
    )
    torch.save(
        [
            torch.full((n_test, 28, 28), 100.0 * scale),
            torch.randint(0, 62, (n_test,)),
            torch.randint(0, n_writers, (n_test,)),
        ],
        processed / "femnist_test.pt",
    )
    torch.save([f"writer_{i}" for i in range(n_writers)], processed / "femnist_user_keys.pt")


class TestWriterPartition:
    def test_clients_are_largest_writers(self) -> None:
        sizes = [60, 50, 40, 30, 20, 15, 12, 11, 10]
        dataset = _make_writer_dataset(sizes, [[0] * n for n in sizes])
        parts = partition_dataset(dataset, 4, "writer", seed=42)
        assert [len(p) for p in parts] == [60, 50, 40, 30]
        for cid, part in enumerate(parts):
            users = set(int(_users_of(dataset)[i]) for i in part.indices)
            assert users == {cid}

    def test_small_writers_merged_into_nearest_label_distribution(self) -> None:
        sizes = [40, 30, 5, 6]
        labels = [[0] * 40, [1] * 30, [0] * 5, [1] * 6]
        dataset = _make_writer_dataset(sizes, labels)
        parts = partition_dataset(dataset, 2, "writer", seed=42)
        assert [len(p) for p in parts] == [45, 36]
        assert {int(_users_of(dataset)[i]) for i in parts[0].indices} == {0, 2}
        assert {int(_users_of(dataset)[i]) for i in parts[1].indices} == {1, 3}

    def test_large_non_client_writers_dropped(self) -> None:
        sizes = [50, 40, 30, 12]
        labels = [[0] * n for n in sizes]
        dataset = _make_writer_dataset(sizes, labels)
        parts = partition_dataset(dataset, 2, "writer", seed=42)
        assert sum(len(p) for p in parts) == 90  # 50 + 40; writers 2, 3 dropped

    def test_merge_threshold_boundary(self) -> None:
        sizes = [50, 10, 9]
        labels = [[0] * n for n in sizes]
        dataset = _make_writer_dataset(sizes, labels)
        parts = partition_dataset(dataset, 1, "writer", seed=42)
        assert len(parts[0]) == 59  # writer with exactly 10 kept out, 9 merged

    def test_deterministic_and_seed_independent(self) -> None:
        sizes = [60, 50, 40, 30, 20, 15, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]
        labels = [[i % 4 for i in range(n)] for n in sizes]
        dataset = _make_writer_dataset(sizes, labels)
        a = partition_dataset(dataset, 6, "writer", seed=1)
        b = partition_dataset(dataset, 6, "writer", seed=999)
        assert [p.indices for p in a] == [p.indices for p in b]

    def test_single_full_parity(self) -> None:
        sizes = [60, 50, 40, 30, 20, 15, 12, 11, 10, 9, 8, 7]
        labels = [[i % 4 for i in range(n)] for n in sizes]
        dataset = _make_writer_dataset(sizes, labels)
        full = partition_dataset(dataset, 5, "writer", seed=42)
        for i in range(5):
            single = partition_single(dataset, 5, i, "writer", seed=42)
            assert single.indices == full[i].indices, i

    def test_min30_not_applied_to_writer(self) -> None:
        sizes = [50, 20, 15, 10]
        labels = [[0] * n for n in sizes]
        dataset = _make_writer_dataset(sizes, labels)
        parts = partition_dataset(dataset, 2, "writer", seed=42, min_samples=30)
        assert [len(p) for p in parts] == [50, 20]

    def test_num_clients_exceeding_writers_raises(self) -> None:
        dataset = _make_writer_dataset([50, 40], [[0] * 50, [0] * 40])
        with pytest.raises(ValueError, match="num_clients"):
            partition_dataset(dataset, 5, "writer", seed=42)

    def test_partition_kwargs(self) -> None:
        assert build_partition_kwargs("writer") == {
            "type": "writer",
            "merge_threshold": 10,
        }


class TestFEMNISTDataset:
    def test_loads_train_split(self, tmp_path: Path) -> None:
        _write_fake_femnist(tmp_path, n_train=100, n_test=20, n_writers=5)
        ds = FEMNISTDataset(str(tmp_path), train=True)
        assert len(ds) == 100
        assert ds.data.shape == (100, 28, 28)
        assert ds.targets.shape == (100,)
        assert int(ds.targets.min()) >= 0 and int(ds.targets.max()) < 62
        assert int(ds.users.min()) >= 0 and int(ds.users.max()) < 5
        assert ds.user_keys == [f"writer_{i}" for i in range(5)]

    def test_loads_test_split(self, tmp_path: Path) -> None:
        _write_fake_femnist(tmp_path, n_train=100, n_test=20, n_writers=5)
        ds = FEMNISTDataset(str(tmp_path), train=False)
        assert len(ds) == 20

    def test_missing_files_raise_clear_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="femnist.tar.gz"):
            FEMNISTDataset(str(tmp_path))

    def test_partial_files_raise(self, tmp_path: Path) -> None:
        _write_fake_femnist(tmp_path)
        (tmp_path / "FEMNIST" / "processed" / "femnist_user_keys.pt").unlink()
        with pytest.raises(FileNotFoundError, match="femnist"):
            FEMNISTDataset(str(tmp_path))

    def test_normalizes_255_scale(self, tmp_path: Path) -> None:
        _write_fake_femnist(tmp_path, scale=1.0)
        ds = FEMNISTDataset(str(tmp_path))
        assert float(ds.data.max()) == pytest.approx(200.0 / 255.0)

    def test_keeps_01_scale(self, tmp_path: Path) -> None:
        _write_fake_femnist(tmp_path, scale=0.001)
        ds = FEMNISTDataset(str(tmp_path))
        assert float(ds.data.max()) == pytest.approx(0.2)

    def test_transform_applied(self, tmp_path: Path) -> None:
        _write_fake_femnist(tmp_path)
        ds = FEMNISTDataset(str(tmp_path), transform=lambda t: t * 2.0)
        img, target = ds[0]
        assert img.shape == (1, 28, 28)
        assert isinstance(target, int)
        assert torch.allclose(img, ds.data[0].unsqueeze(0) * 2.0)

    def test_counts(self, tmp_path: Path) -> None:
        _write_fake_femnist(tmp_path, n_train=654, n_test=163, n_writers=35)
        assert femnist_counts(str(tmp_path)) == (654, 163, 35)


def test_femnist_registry_and_meta() -> None:
    assert DATASET_REGISTRY["femnist"] is FEMNISTDataset
    assert NUM_CLASSES_MAP["femnist"] == 62
    assert TRANSFORMS_MAP["femnist"] is not None
    assert get_num_classes("femnist") == 62
