from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from src.config.loader import BOConfig, PersonalizationConfig
from src.privacy.accountant import RDPAccountant
from src.privacy.personalization import (
    _compute_label_entropy,
    _get_num_classes,
    _get_targets,
    assign_epsilon_bounds,
    compute_budget_weight,
)


def _make_dataset(labels: list[int]) -> TensorDataset:
    n = len(labels)
    features = torch.randn(n, 3, 8, 8)
    targets = torch.tensor(labels)
    return TensorDataset(features, targets)


def test_weight_custom_strategy() -> None:
    config = PersonalizationConfig(
        enabled=True,
        strategy="custom",
        client_epsilon_map={0: 1.0, 1: 2.5, 2: 5.0},
    )
    dataset = _make_dataset([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    assert compute_budget_weight(0, dataset, config) == 1.0
    assert compute_budget_weight(1, dataset, config) == 2.5
    assert compute_budget_weight(2, dataset, config) == 5.0


def test_weight_custom_missing_key() -> None:
    config = PersonalizationConfig(
        enabled=True,
        strategy="custom",
        client_epsilon_map={0: 1.0},
    )
    dataset = _make_dataset([0, 1, 2])

    with pytest.raises(ValueError, match="not found in client_epsilon_map"):
        compute_budget_weight(5, dataset, config)


def test_weight_uniform_strategy() -> None:
    config = PersonalizationConfig(
        enabled=True,
        strategy="uniform",
    )
    dataset = _make_dataset([0, 1, 2])

    weights = [compute_budget_weight(0, dataset, config) for _ in range(100)]

    assert all(0.0 <= w <= 1.0 for w in weights)
    assert len(set(weights)) > 1


def test_weight_data_proportional_strategy() -> None:
    total_size = 10 + 100  # small + large
    config = PersonalizationConfig(
        enabled=True,
        strategy="data_proportional",
    )

    small_dataset = _make_dataset([0, 1] * 5)
    large_dataset = _make_dataset([0, 1] * 50)

    small_weight = compute_budget_weight(0, small_dataset, config, num_clients=2, total_train_size=total_size)
    large_weight = compute_budget_weight(0, large_dataset, config, num_clients=2, total_train_size=total_size)

    assert small_weight < large_weight


def test_weight_heterogeneity_strategy() -> None:
    config = PersonalizationConfig(
        enabled=True,
        strategy="heterogeneity",
    )

    uniform_dataset = _make_dataset([0, 1, 2, 3] * 25)
    single_class_dataset = _make_dataset([0] * 100)

    uniform_weight = compute_budget_weight(0, uniform_dataset, config)
    single_class_weight = compute_budget_weight(0, single_class_dataset, config)

    assert uniform_weight < single_class_weight


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
        accountant.step(sigma=1.0, clipping_norm=1.0)

    state = accountant.get_state()
    restored = RDPAccountant.from_state(state)

    assert restored.get_epsilon() == pytest.approx(accountant.get_epsilon())
    assert restored.total_steps() == accountant.total_steps()


def test_rdp_accountant_serialization_empty() -> None:
    accountant = RDPAccountant(delta=1e-6)
    state = accountant.get_state()
    restored = RDPAccountant.from_state(state)

    assert restored.get_epsilon() == 0.0
    assert restored.total_steps() == 0


def _make_bo_config(**kwargs) -> BOConfig:
    defaults = dict(
        enabled=True,
        warmup_rounds=20,
        epsilon_min=0.1,
        epsilon_max=10.0,
        epsilon_budget=10.0,
        bounds_strategy="global",
        bounds_ratio_min=0.1,
        bounds_ratio_max=1.0,
    )
    defaults.update(kwargs)
    return BOConfig(**defaults)


def _make_personalization_config(**kwargs) -> PersonalizationConfig:
    defaults = dict(
        enabled=True,
        strategy="custom",
        client_epsilon_map={0: 5.0, 1: 2.0},
        epsilon_min=0.1,
        epsilon_max=10.0,
    )
    defaults.update(kwargs)
    return PersonalizationConfig(**defaults)


class TestAssignEpsilonBounds:
    def test_global_strategy(self) -> None:
        pc = _make_personalization_config(enabled=False)
        bc = _make_bo_config(bounds_strategy="global", epsilon_min=0.5, epsilon_max=8.0)
        dataset = _make_dataset([0, 1, 2])
        eps_min, eps_max, warmup = assign_epsilon_bounds(0, dataset, pc, bc)
        assert eps_min == 0.5
        assert eps_max == 8.0
        assert warmup == 20

    def test_custom_map_strategy(self) -> None:
        pc = _make_personalization_config(enabled=False)
        bc = _make_bo_config(
            bounds_strategy="custom_map",
            client_eps_min_map={0: 0.5, 1: 1.0},
            client_eps_max_map={0: 5.0, 1: 10.0},
        )
        dataset = _make_dataset([0, 1, 2])
        eps_min_0, eps_max_0, _ = assign_epsilon_bounds(0, dataset, pc, bc)
        eps_min_1, eps_max_1, _ = assign_epsilon_bounds(1, dataset, pc, bc)
        assert eps_min_0 == 0.5
        assert eps_max_0 == 5.0
        assert eps_min_1 == 1.0
        assert eps_max_1 == 10.0

    def test_custom_map_missing_key(self) -> None:
        pc = _make_personalization_config(enabled=False)
        bc = _make_bo_config(
            bounds_strategy="custom_map",
            client_eps_min_map={0: 0.5},
            client_eps_max_map={0: 5.0},
        )
        dataset = _make_dataset([0, 1, 2])
        with pytest.raises(ValueError, match="not found in client_eps_min_map"):
            assign_epsilon_bounds(5, dataset, pc, bc)

    def test_custom_map_string_keys(self) -> None:
        pc = _make_personalization_config(enabled=False)
        bc = _make_bo_config(
            bounds_strategy="custom_map",
            client_eps_min_map={"0": 0.5, "1": 1.0},
            client_eps_max_map={"0": 5.0, "1": 10.0},
        )
        dataset = _make_dataset([0, 1, 2])
        eps_min_0, eps_max_0, _ = assign_epsilon_bounds(0, dataset, pc, bc)
        eps_min_1, eps_max_1, _ = assign_epsilon_bounds(1, dataset, pc, bc)
        assert eps_min_0 == 0.5
        assert eps_max_0 == 5.0
        assert eps_min_1 == 1.0
        assert eps_max_1 == 10.0

    def test_warmup_string_keys(self) -> None:
        pc = _make_personalization_config(enabled=False)
        bc = _make_bo_config(
            bounds_strategy="global",
            warmup_rounds=20,
            client_warmup_rounds_map={"0": 15, "5": 25},
        )
        dataset = _make_dataset([0, 1, 2])
        _, _, w0 = assign_epsilon_bounds(0, dataset, pc, bc)
        _, _, w1 = assign_epsilon_bounds(1, dataset, pc, bc)
        _, _, w5 = assign_epsilon_bounds(5, dataset, pc, bc)
        assert w0 == 15
        assert w1 == 20
        assert w5 == 25

    def test_from_epsilon_custom_strategy(self) -> None:
        pc = _make_personalization_config(
            strategy="custom",
            client_epsilon_map={0: 10.0},
        )
        bc = _make_bo_config(
            bounds_strategy="from_epsilon",
            bounds_ratio_min=0.1,
            bounds_ratio_max=1.0,
        )
        dataset = _make_dataset([0, 1, 2])
        eps_min, eps_max, warmup = assign_epsilon_bounds(0, dataset, pc, bc)
        assert eps_min == pytest.approx(1.0)
        assert eps_max == pytest.approx(10.0)
        assert warmup == 20

    def test_from_epsilon_data_proportional(self) -> None:
        pc = _make_personalization_config(
            strategy="data_proportional",
            epsilon_min=0.1,
            epsilon_max=10.0,
        )
        bc = _make_bo_config(
            bounds_strategy="from_epsilon",
            bounds_ratio_min=0.2,
            bounds_ratio_max=2.0,
        )
        small = _make_dataset([0] * 10)
        eps_min, eps_max, _ = assign_epsilon_bounds(0, small, pc, bc, num_clients=10)
        # weight = expected_per_client / client_size = (10*10/10) / 10 = 1.0
        # eps_min = max(1.0 * 0.2, 1e-6) = 0.2
        # eps_max = 1.0 * 2.0 = 2.0
        assert eps_min == pytest.approx(0.2)
        assert eps_max == pytest.approx(2.0)

    def test_from_epsilon_no_personalization_raises(self) -> None:
        pc = _make_personalization_config(enabled=False)
        bc = _make_bo_config(bounds_strategy="from_epsilon")
        dataset = _make_dataset([0, 1, 2])
        with pytest.raises(ValueError, match="personalization.enabled=True"):
            assign_epsilon_bounds(0, dataset, pc, bc)

    def test_warmup_per_client_override(self) -> None:
        pc = _make_personalization_config(enabled=False)
        bc = _make_bo_config(
            bounds_strategy="global",
            warmup_rounds=20,
            client_warmup_rounds_map={0: 15, 5: 25},
        )
        dataset = _make_dataset([0, 1, 2])
        _, _, w0 = assign_epsilon_bounds(0, dataset, pc, bc)
        _, _, w1 = assign_epsilon_bounds(1, dataset, pc, bc)
        _, _, w5 = assign_epsilon_bounds(5, dataset, pc, bc)
        assert w0 == 15
        assert w1 == 20
        assert w5 == 25

    def test_from_epsilon_ratio_not_budget(self) -> None:
        pc = _make_personalization_config(
            strategy="custom",
            client_epsilon_map={0: 4.0},
        )
        bc = _make_bo_config(
            bounds_strategy="from_epsilon",
            bounds_ratio_min=0.25,
            bounds_ratio_max=0.75,
        )
        dataset = _make_dataset([0, 1, 2])
        eps_min, eps_max, _ = assign_epsilon_bounds(0, dataset, pc, bc)
        assert eps_min == pytest.approx(1.0)
        assert eps_max == pytest.approx(3.0)

    def test_unknown_strategy_raises(self) -> None:
        pc = _make_personalization_config(enabled=False)
        bc = _make_bo_config(bounds_strategy="unknown")
        dataset = _make_dataset([0, 1, 2])
        with pytest.raises(ValueError, match="Unknown bounds_strategy"):
            assign_epsilon_bounds(0, dataset, pc, bc)
