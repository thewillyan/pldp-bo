from __future__ import annotations

from functools import partial
from typing import Any

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.config.loader import DataConfig
from src.data.femnist import FEMNISTDataset

DATASET_REGISTRY: dict[str, type[Dataset[Any]]] = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "mnist": datasets.MNIST,
    "femnist": FEMNISTDataset,
}

CIFAR10_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

CIFAR100_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
)

MNIST_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

FEMNIST_TRANSFORMS = transforms.Compose(
    [
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

TRANSFORMS_MAP = {
    "cifar10": CIFAR10_TRANSFORMS,
    "cifar100": CIFAR100_TRANSFORMS,
    "mnist": MNIST_TRANSFORMS,
    "femnist": FEMNIST_TRANSFORMS,
}

NUM_CLASSES_MAP = {
    "cifar10": 10,
    "cifar100": 100,
    "mnist": 10,
    "femnist": 62,
}


def get_dataset_kwargs(config: DataConfig) -> dict[str, object]:
    transform = TRANSFORMS_MAP.get(config.name, transforms.ToTensor())
    return {"root": config.data_dir, "transform": transform, "download": True}


def _worker_init_fn(worker_id: int, base_seed: int) -> None:
    import numpy as np
    import torch

    worker_seed = base_seed + worker_id
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloaders(
    subset: Dataset[Any] | Subset[Any],
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> DataLoader[Any]:
    kwargs: dict[str, Any] = {"batch_size": batch_size, "shuffle": shuffle}
    if seed is not None and shuffle:
        kwargs["worker_init_fn"] = partial(_worker_init_fn, base_seed=seed)
    return DataLoader(subset, **kwargs)


def get_num_classes(dataset_name: str) -> int:
    return NUM_CLASSES_MAP.get(dataset_name, 10)
