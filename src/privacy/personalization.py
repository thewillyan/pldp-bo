from __future__ import annotations

import logging
import math

import numpy as np
from torch.utils.data import Dataset, Subset

from src.config.loader import BOConfig, PersonalizationConfig

logger = logging.getLogger(__name__)


def compute_budget_weight(
    partition_id: int,
    train_dataset: Dataset | Subset,
    config: PersonalizationConfig,
    num_clients: int = 1,
    total_train_size: int | None = None,
    rng: np.random.RandomState | None = None,
) -> float:
    strategy = config.strategy
    if strategy == "custom":
        return _weight_custom(partition_id, config)
    if strategy == "data_proportional":
        if total_train_size is None:
            raise ValueError(
                "total_train_size is required for data_proportional strategy. "
                "This should be provided by the caller (e.g., via create_client_dataloader)."
            )
        return _weight_data_proportional(train_dataset, num_clients, total_train_size=total_train_size)
    if strategy == "heterogeneity":
        return _weight_heterogeneity(train_dataset)
    if strategy == "uniform":
        return _weight_uniform(rng=rng)
    raise ValueError(f"Unknown personalization strategy: {strategy}")


def _weight_uniform(
    rng: np.random.RandomState | None = None,
) -> float:
    if rng is None:
        rng = np.random.RandomState()
    return float(rng.uniform(0, 1))


def _weight_custom(partition_id: int, config: PersonalizationConfig) -> float:
    weight_map = {int(k): v for k, v in config.client_epsilon_map.items()}
    if partition_id not in weight_map:
        raise ValueError(
            f"partition_id {partition_id} not found in client_epsilon_map. "
            f"Available: {sorted(weight_map.keys())}",
        )
    return float(weight_map[partition_id])


def _weight_data_proportional(
    dataset: Dataset | Subset, num_clients: int,
    total_train_size: int,
) -> float:
    """Budget weight proportional to data size.

    Clients with more data receive a larger budget weight (weaker
    privacy / less noise), while clients with fewer data receive a
    smaller weight (stronger privacy / more noise).
    """
    client_size = len(dataset)
    expected_per_client = total_train_size / num_clients
    return client_size / expected_per_client


def _weight_heterogeneity(
    dataset: Dataset | Subset,
) -> float:
    entropy = _compute_label_entropy(dataset)
    normalized_entropy = entropy / math.log(max(_get_num_classes(dataset), 2)) if entropy > 0 else 0.0
    return 1.0 - normalized_entropy


def _compute_label_entropy(dataset: Dataset | Subset) -> float:
    targets = _get_targets(dataset)
    class_counts = np.bincount(targets, minlength=_get_num_classes(dataset)).astype(float)
    class_counts = class_counts[class_counts > 0]
    probs = class_counts / class_counts.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def _get_targets(dataset: Dataset | Subset) -> np.ndarray:
    indices = None
    while isinstance(dataset, Subset):
        if indices is None:
            indices = np.asarray(dataset.indices)
        else:
            indices = np.asarray(dataset.indices)[indices]
        dataset = dataset.dataset
    try:
        targets = np.array(dataset.targets)
    except AttributeError:
        targets = dataset.tensors[1].numpy()
    if indices is not None:
        targets = targets[indices]
    return targets


def _get_num_classes(dataset: Dataset | Subset) -> int:
    targets = _get_targets(dataset)
    return len(np.unique(targets))


def _resolve_warmup(
    partition_id: int,
    bo_config: BOConfig,
    num_clients: int,
    num_rounds: int,
    weight: float | None = None,
) -> int:
    warmup_map = {int(k): v for k, v in bo_config.client_warmup_rounds_map.items()}
    if partition_id in warmup_map:
        return warmup_map[partition_id]
    if bo_config.max_warmup_ratio > 0 and weight is not None:
        max_warmup = max(bo_config.min_warmup, math.ceil(bo_config.max_warmup_ratio * num_rounds))
        normalized = min(weight / max(1, num_clients), 1.0)
        return max(bo_config.min_warmup, math.ceil(max_warmup * normalized))
    return bo_config.min_warmup


def assign_epsilon_bounds(
    partition_id: int,
    train_dataset: Dataset | Subset,
    personalization_config: PersonalizationConfig,
    bo_config: BOConfig,
    num_clients: int = 1,
    total_train_size: int | None = None,
    num_rounds: int = 50,
) -> tuple[float, float, int]:
    strategy = bo_config.bounds_strategy
    if strategy == "global":
        warmup = _resolve_warmup(partition_id, bo_config, num_clients, num_rounds, weight=float(num_clients))
        return bo_config.epsilon_min, bo_config.epsilon_max, warmup

    if strategy == "custom_map":
        eps_min_map = {int(k): v for k, v in bo_config.client_eps_min_map.items()}
        eps_max_map = {int(k): v for k, v in bo_config.client_eps_max_map.items()}
        eps_min = eps_min_map.get(partition_id)
        eps_max = eps_max_map.get(partition_id)
        if eps_min is None or eps_max is None:
            raise ValueError(
                f"partition_id {partition_id} not found in client_eps_min_map "
                f"or client_eps_max_map. "
                f"Available eps_min keys: {sorted(eps_min_map.keys())}, "
                f"eps_max keys: {sorted(eps_max_map.keys())}",
            )
        weight = eps_max / max(bo_config.epsilon_max, 1e-12)
        warmup = _resolve_warmup(partition_id, bo_config, num_clients, num_rounds, weight)
        return eps_min, eps_max, warmup

    if strategy == "from_epsilon":
        if not personalization_config.enabled:
            raise ValueError(
                "bounds_strategy='from_epsilon' requires personalization.enabled=True",
            )
        weight = compute_budget_weight(
            partition_id, train_dataset, personalization_config, num_clients,
            total_train_size=total_train_size,
        )
        eps_min = max(weight * bo_config.bounds_ratio_min, 1e-6)
        eps_max = weight * bo_config.bounds_ratio_max
        warmup = _resolve_warmup(partition_id, bo_config, num_clients, num_rounds, weight)
        return eps_min, eps_max, warmup

    raise ValueError(f"Unknown bounds_strategy: {strategy}")
