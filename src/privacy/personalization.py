from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

import logging

from src.config.loader import BOConfig, PersonalizationConfig

logger = logging.getLogger(__name__)


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


def assign_epsilon_bounds(
    partition_id: int,
    train_dataset: Union[Dataset, Subset],
    personalization_config: PersonalizationConfig,
    bo_config: BOConfig,
    num_clients: int = 1,
) -> tuple[float, float, int]:
    warmup = bo_config.client_warmup_rounds_map.get(partition_id, bo_config.warmup_rounds)

    strategy = bo_config.bounds_strategy
    if strategy == "global":
        return bo_config.epsilon_min, bo_config.epsilon_max, warmup

    if strategy == "custom_map":
        eps_min = bo_config.client_eps_min_map.get(partition_id)
        eps_max = bo_config.client_eps_max_map.get(partition_id)
        if eps_min is None or eps_max is None:
            raise ValueError(
                f"partition_id {partition_id} not found in client_eps_min_map "
                f"or client_eps_max_map. "
                f"Available eps_min keys: {sorted(bo_config.client_eps_min_map.keys())}, "
                f"eps_max keys: {sorted(bo_config.client_eps_max_map.keys())}"
            )
        return eps_min, eps_max, warmup

    if strategy == "from_epsilon":
        if not personalization_config.enabled:
            raise ValueError(
                "bounds_strategy='from_epsilon' requires personalization.enabled=True"
            )
        epsilon = assign_epsilon(
            partition_id, train_dataset, personalization_config, num_clients,
        )
        eps_min = max(epsilon * bo_config.bounds_ratio_min, 1e-6)
        eps_max = epsilon * bo_config.bounds_ratio_max
        return eps_min, eps_max, warmup

    raise ValueError(f"Unknown bounds_strategy: {strategy}")
