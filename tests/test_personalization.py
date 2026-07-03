from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from src.config.loader import PersonalizationConfig
from src.privacy.personalization import (
    _compute_label_entropy,
    _get_num_classes,
    _get_targets,
    assign_epsilon,
)
from src.privacy.accountant import RDPAccountant


def _make_dataset(labels: list[int]) -> TensorDataset:
    n = len(labels)
    features = torch.randn(n, 3, 8, 8)
    targets = torch.tensor(labels)
    return TensorDataset(features, targets)


def test_assign_custom_strategy() -> None:
    config = PersonalizationConfig(
        enabled=True,
        strategy="custom",
        client_epsilon_map={0: 1.0, 1: 2.5, 2: 5.0},
    )
    dataset = _make_dataset([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    assert assign_epsilon(0, dataset, config) == 1.0
    assert assign_epsilon(1, dataset, config) == 2.5
    assert assign_epsilon(2, dataset, config) == 5.0


def test_assign_custom_missing_key() -> None:
    config = PersonalizationConfig(
        enabled=True,
        strategy="custom",
        client_epsilon_map={0: 1.0},
    )
    dataset = _make_dataset([0, 1, 2])

    with pytest.raises(ValueError, match="not found in client_epsilon_map"):
        assign_epsilon(5, dataset, config)


def test_assign_uniform_strategy() -> None:
    config = PersonalizationConfig(
        enabled=True,
        strategy="uniform",
        epsilon_min=1.0,
        epsilon_max=10.0,
    )
    dataset = _make_dataset([0, 1, 2])

    np.random.seed(42)
    epsilons = [assign_epsilon(0, dataset, config) for _ in range(100)]

    assert all(1.0 <= e <= 10.0 for e in epsilons)
    assert len(set(epsilons)) > 1


def test_assign_data_proportional_strategy() -> None:
    total_size = 10 + 100  # small + large
    config = PersonalizationConfig(
        enabled=True,
        strategy="data_proportional",
        epsilon_base=5.0,
        epsilon_min=0.1,
        epsilon_max=10.0,
        client_epsilon_map={"__total_size": total_size},
    )

    small_dataset = _make_dataset([0, 1] * 5)
    large_dataset = _make_dataset([0, 1] * 50)

    small_eps = assign_epsilon(0, small_dataset, config, num_clients=2)
    large_eps = assign_epsilon(0, large_dataset, config, num_clients=2)

    assert small_eps > large_eps


def test_data_proportional_respects_bounds() -> None:
    config = PersonalizationConfig(
        enabled=True,
        strategy="data_proportional",
        epsilon_base=5.0,
        epsilon_min=1.0,
        epsilon_max=8.0,
        client_epsilon_map={"__total_size": 10100},
    )

    tiny_dataset = _make_dataset([0])
    huge_dataset = _make_dataset([0] * 10000)

    tiny_eps = assign_epsilon(0, tiny_dataset, config, num_clients=100)
    huge_eps = assign_epsilon(0, huge_dataset, config, num_clients=100)

    assert tiny_eps >= 1.0
    assert tiny_eps <= 8.0
    assert huge_eps >= 1.0
    assert huge_eps <= 8.0


def test_assign_heterogeneity_strategy() -> None:
    config = PersonalizationConfig(
        enabled=True,
        strategy="heterogeneity",
        epsilon_min=0.5,
        epsilon_max=10.0,
    )

    uniform_dataset = _make_dataset([0, 1, 2, 3] * 25)
    single_class_dataset = _make_dataset([0] * 100)

    uniform_eps = assign_epsilon(0, uniform_dataset, config)
    single_class_eps = assign_epsilon(0, single_class_dataset, config)

    assert uniform_eps != single_class_eps


def test_heterogeneity_respects_bounds() -> None:
    config = PersonalizationConfig(
        enabled=True,
        strategy="heterogeneity",
        epsilon_min=2.0,
        epsilon_max=7.0,
    )

    dataset = _make_dataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10)
    eps = assign_epsilon(0, dataset, config)

    assert eps >= 2.0
    assert eps <= 7.0


def test_compute_label_entropy_uniform() -> None:
    dataset = _make_dataset([0, 1, 2, 3] * 25)
    entropy = _compute_label_entropy(dataset)
    expected = np.log(4)
    assert abs(entropy - expected) < 1e-6


def test_compute_label_entropy_single_class() -> None:
    dataset = _make_dataset([0] * 100)
    entropy = _compute_label_entropy(dataset)
    assert entropy == pytest.approx(0.0)


def test_get_targets_from_subset() -> None:
    from torch.utils.data import Subset

    dataset = _make_dataset([0, 1, 2, 3, 4])
    subset = Subset(dataset, [0, 2, 4])
    targets = _get_targets(subset)
    np.testing.assert_array_equal(targets, [0, 2, 4])


def test_get_num_classes() -> None:
    dataset = _make_dataset([0, 1, 2, 0, 1, 2])
    assert _get_num_classes(dataset) == 3


def test_rdp_accountant_serialization() -> None:
    accountant = RDPAccountant(delta=1e-5)
    for _ in range(5):
        accountant.step(noise_multiplier=1.0, sample_rate=0.01)

    state = accountant.get_state()
    restored = RDPAccountant.from_state(state)

    assert restored.get_epsilon() == pytest.approx(accountant.get_epsilon())
    assert restored.total_steps() == accountant.total_steps()
    assert restored._delta == accountant._delta


def test_rdp_accountant_serialization_empty() -> None:
    accountant = RDPAccountant(delta=1e-6)
    state = accountant.get_state()
    restored = RDPAccountant.from_state(state)

    assert restored.get_epsilon() == 0.0
    assert restored.total_steps() == 0
    assert restored._delta == 1e-6
