from __future__ import annotations

from collections.abc import Sized
from functools import lru_cache
from typing import Any, cast

from torch.utils.data import DataLoader, Dataset, Subset

from src.config.loader import DataConfig
from src.data.dataloaders import (
    DATASET_REGISTRY,
    TRANSFORMS_MAP,
    create_dataloaders,
)
from src.data.partitioner import partition_single, split_holdout


def create_dataset(config: DataConfig, train: bool = True) -> Dataset[Any]:
    """Return the full official train (or test) split; no hold-out applied."""
    dataset_cls = cast(Any, DATASET_REGISTRY[config.name])
    transform = TRANSFORMS_MAP.get(config.name)

    full_dataset: Dataset[Any] = dataset_cls(
        root=config.data_dir,
        train=train,
        transform=transform,
        download=True,
    )
    return full_dataset


@lru_cache(maxsize=2)
def _cached_dataset(name: str, data_dir: str, train: bool = True) -> Dataset[Any]:
    """Load the dataset once and cache it by (name, data_dir, train).

    This avoids each client in a simulation loading the full dataset
    independently; train and test splits fit the two cache slots.
    """
    cfg = DataConfig(name=name, data_dir=data_dir)
    return create_dataset(cfg, train=train)


def create_test_dataset(config: DataConfig) -> Dataset[Any]:
    """Return the full official test split (no hold-out)."""
    return _cached_dataset(config.name, config.data_dir, train=False)


def create_test_loader(config: DataConfig) -> DataLoader[Any]:
    """Return a non-shuffled dataloader over the official test split."""
    return create_dataloaders(create_test_dataset(config), config.batch_size, shuffle=False)


def create_client_dataloader(
    config: DataConfig,
    partition_id: int,
    num_partitions: int,
    seed: int = 42,
) -> tuple[DataLoader[Any], DataLoader[Any], Subset[Any], Subset[Any], int]:
    """Create dataloaders for a single client partition.

    Only loads the specific partition for the given client, not all partitions.
    Each client holds out a fixed validation_frac of *its own* partition
    (seeded by seed + partition_id, so the split is identical across rounds);
    the hold-out data is never seen in training. Returns
    (train_loader, val_loader, train_subset, val_subset, total_train_size).
    """
    full_dataset = _cached_dataset(config.name, config.data_dir)

    client_subset = partition_single(
        full_dataset,
        num_partitions,
        partition_id,
        config.partition_type,
        config.partition_alpha,
        seed=seed,
        min_samples=config.partition_min_samples,
    )

    train_subset, val_subset = split_holdout(
        client_subset,
        config.val_split,
        seed + partition_id,
    )

    train_loader = create_dataloaders(train_subset, config.batch_size, shuffle=True, seed=seed)
    val_loader = create_dataloaders(val_subset, config.batch_size, shuffle=False)

    total_train_size = len(cast(Sized, full_dataset))
    return train_loader, val_loader, train_subset, val_subset, total_train_size
