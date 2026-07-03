from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from src.config.loader import PersonalizationConfig


def assign_epsilon(
    partition_id: int,
    train_dataset: Union[Dataset, Subset],
    config: PersonalizationConfig,
    num_clients: int = 1,
) -> float:
    strategy = config.strategy
    if strategy == "custom":
        return _assign_custom(partition_id, config)
    elif strategy == "data_proportional":
        return _assign_data_proportional(train_dataset, config, num_clients)
    elif strategy == "heterogeneity":
        return _assign_heterogeneity(train_dataset, config)
    elif strategy == "uniform":
        return _assign_uniform(config)
    else:
        raise ValueError(f"Unknown personalization strategy: {strategy}")


def _assign_uniform(config: PersonalizationConfig) -> float:
    return float(np.random.uniform(config.epsilon_min, config.epsilon_max))


def _assign_custom(partition_id: int, config: PersonalizationConfig) -> float:
    epsilon_map = {int(k): v for k, v in config.client_epsilon_map.items()}
    if partition_id not in epsilon_map:
        raise ValueError(
            f"partition_id {partition_id} not found in client_epsilon_map. "
            f"Available: {sorted(epsilon_map.keys())}"
        )
    return float(epsilon_map[partition_id])


def _assign_data_proportional(
    dataset: Union[Dataset, Subset], config: PersonalizationConfig, num_clients: int
) -> float:
    client_size = len(dataset)
    total_size = config.client_epsilon_map.get("__total_size")
    if total_size is None:
        total_size = client_size * num_clients
    expected_per_client = total_size / num_clients

    epsilon = config.epsilon_base * (expected_per_client / client_size)
    return float(np.clip(epsilon, config.epsilon_min, config.epsilon_max))


def _assign_heterogeneity(
    dataset: Union[Dataset, Subset], config: PersonalizationConfig
) -> float:
    entropy = _compute_label_entropy(dataset)
    normalized_entropy = entropy / math.log(_get_num_classes(dataset)) if entropy > 0 else 0.0

    range_span = config.epsilon_max - config.epsilon_min
    epsilon = config.epsilon_min + range_span * (1.0 - normalized_entropy)
    return float(np.clip(epsilon, config.epsilon_min, config.epsilon_max))


def _compute_label_entropy(dataset: Union[Dataset, Subset]) -> float:
    targets = _get_targets(dataset)
    class_counts = np.bincount(targets, minlength=_get_num_classes(dataset)).astype(float)
    class_counts = class_counts[class_counts > 0]
    probs = class_counts / class_counts.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def _get_targets(dataset: Union[Dataset, Subset]) -> np.ndarray:
    if isinstance(dataset, Subset):
        if hasattr(dataset.dataset, "targets"):
            all_targets = np.array(dataset.dataset.targets)
            return all_targets[dataset.indices]
        elif hasattr(dataset.dataset, "tensors"):
            return dataset.dataset.tensors[1].numpy()[dataset.indices]
    elif hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    elif hasattr(dataset, "tensors"):
        return dataset.tensors[1].numpy()
    raise ValueError("Cannot extract targets from dataset")


def _get_num_classes(dataset: Union[Dataset, Subset]) -> int:
    targets = _get_targets(dataset)
    return int(len(np.unique(targets)))
