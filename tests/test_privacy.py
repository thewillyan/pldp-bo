from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.loader import ExperimentConfig
from src.privacy import RDPAccountant, simulate_epsilon
from src.privacy.constants import RDP_ALPHAS
from src.privacy.metrics import compute_utility_loss
from src.privacy.per_update_dp import (
    PerUpdateGaussianMechanism,
    _clip_update,
    add_gaussian_noise,
    calibrate_sigma,
    calibrate_sigma_dp_sgd,
    compute_rdp_cost,
    compute_rdp_cost_dp_sgd,
    enforce_epsilon_budget,
)


def test_rdp_accountant_initial() -> None:
    accountant = RDPAccountant(delta=1e-5)
    eps = accountant.get_epsilon()
    assert eps == 0.0


def test_rdp_accountant_steps() -> None:
    accountant = RDPAccountant(delta=1e-5)
    for _ in range(10):
        accountant.step(sigma=1.0, clipping_norm=1.0)
    eps = accountant.get_epsilon()
    assert eps > 0


def test_rdp_accountant_privacy_spent() -> None:
    accountant = RDPAccountant(delta=1e-5)
    for _ in range(5):
        accountant.step(sigma=1.0, clipping_norm=1.0)
    spent = accountant.get_privacy_spent()
    assert "epsilon" in spent
    assert "delta" in spent
    assert spent["delta"] == 1e-5


def test_simulate_epsilon() -> None:
    epsilons = simulate_epsilon(
        num_rounds=10,
        sigma=1.0,
        clipping_norm=1.0,
    )
    assert len(epsilons) == 10
    assert epsilons[-1] > epsilons[0]


def test_calibrate_sigma() -> None:
    sigma = calibrate_sigma(epsilon=1.0, clipping_norm=1.0, delta=1e-5)
    assert sigma > 0
    assert np.isfinite(sigma)


def test_calibrate_sigma_larger_epsilon_gives_smaller_sigma() -> None:
    sigma_low = calibrate_sigma(epsilon=0.5, clipping_norm=1.0, delta=1e-5)
    sigma_high = calibrate_sigma(epsilon=2.0, clipping_norm=1.0, delta=1e-5)
    assert sigma_low > sigma_high


def test_calibrate_sigma_no_default_floor() -> None:
    sigma = calibrate_sigma(epsilon=100.0, clipping_norm=1.0, delta=1e-5)
    assert sigma < 1.0
    assert sigma > 0
    sigma_large = calibrate_sigma(epsilon=1e9, clipping_norm=5.0, delta=1e-5)
    assert sigma_large < 5.0
    assert sigma_large > 0


def test_calibrate_sigma_invalid_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon must be positive"):
        calibrate_sigma(epsilon=0.0, clipping_norm=1.0, delta=1e-5)


def test__clip_update_below_threshold() -> None:
    delta = np.array([0.1, 0.2, 0.3])
    clipped = _clip_update(delta, clipping_norm=1.0)
    np.testing.assert_array_almost_equal(clipped, delta)


def test__clip_update_above_threshold() -> None:
    delta = np.array([3.0, 4.0])  # norm = 5.0
    clipped = _clip_update(delta, clipping_norm=1.0)
    expected_norm = 1.0
    clipped_norm = np.linalg.norm(clipped)
    assert abs(clipped_norm - expected_norm) < 1e-6


def test_add_gaussian_noise_shape() -> None:
    delta = np.ones(100, dtype=np.float64)
    noisy = add_gaussian_noise(delta, sigma=0.5)
    assert noisy.shape == delta.shape
    assert noisy.dtype == delta.dtype


def test_add_gaussian_noise_statistical() -> None:
    rng = np.random.RandomState(42)
    original = rng.randn(10000).astype(np.float64)
    sigma = 1.0
    noisy = add_gaussian_noise(original, sigma)
    noise = noisy - original
    assert abs(np.std(noise) - sigma) < 0.1


def test_per_update_gaussian_mechanism_apply() -> None:
    mechanism = PerUpdateGaussianMechanism(clipping_norm=1.0, delta=1e-5)
    delta = np.array([5.0, 0.0])  # norm = 5.0, will be clipped
    noisy, sigma = mechanism.apply(delta, epsilon=1.0)
    assert sigma > 0
    assert noisy.shape == delta.shape
    clipped_norm = np.linalg.norm(_clip_update(delta, 1.0))
    assert abs(clipped_norm - 1.0) < 1e-6


def test_compute_rdp_cost() -> None:
    cost = compute_rdp_cost(alpha=2.0, sigma=1.0, clipping_norm=1.0)
    assert cost == pytest.approx(1.0)
    cost_double = compute_rdp_cost(alpha=2.0, sigma=2.0, clipping_norm=1.0)
    assert cost_double == pytest.approx(0.25)


def test_compute_rdp_cost_invalid_clipping_norm() -> None:
    with pytest.raises(ValueError, match="clipping_norm must be positive"):
        compute_rdp_cost(alpha=2.0, sigma=1.0, clipping_norm=-1.0)
    with pytest.raises(ValueError, match="clipping_norm must be positive"):
        compute_rdp_cost(alpha=2.0, sigma=1.0, clipping_norm=0.0)


def test_rdp_accountant_get_state() -> None:
    accountant = RDPAccountant(delta=1e-5)
    for _ in range(3):
        accountant.step(sigma=0.5, clipping_norm=1.0)

    state = accountant.get_state()
    assert "delta" in state
    assert "steps" in state
    import json
    assert len(json.loads(state["steps"])) == 3
    assert "rdp_per_alpha" in state


def test_rdp_accountant_from_state_roundtrip() -> None:
    original = RDPAccountant(delta=1e-5)
    for _ in range(5):
        original.step(sigma=1.0, clipping_norm=1.0)

    state = original.get_state()
    restored = RDPAccountant.from_state(state)

    assert restored.get_epsilon() == pytest.approx(original.get_epsilon())
    assert restored.total_steps() == original.total_steps()


def test_rdp_accountant_from_state_empty() -> None:
    state = {"delta": 1e-6, "steps": "[]", "rdp_per_alpha": []}
    accountant = RDPAccountant.from_state(state)

    assert accountant.get_epsilon() == 0.0
    assert accountant.total_steps() == 0


def test_rdp_accountant_reset() -> None:
    accountant = RDPAccountant(delta=1e-5)
    for _ in range(5):
        accountant.step(sigma=1.0, clipping_norm=1.0)

    assert accountant.get_epsilon() > 0
    accountant.reset()
    assert accountant.get_epsilon() == 0.0
    assert accountant.total_steps() == 0


def test_rdp_accountant_get_epsilon_infinite_when_no_valid() -> None:
    accountant = RDPAccountant(delta=1e-5)
    accountant._rdp_per_alpha = np.full_like(accountant._rdp_per_alpha, float("inf"))
    accountant._steps.append({"sigma": 1.0, "clipping_norm": 1.0, "num_steps": 1})
    eps = accountant.get_epsilon()
    assert eps == float("inf")
    eps_diag, alpha_diag = accountant.get_epsilon_with_diagnostics()
    assert eps_diag == float("inf")
    assert alpha_diag == 0.0


def test_rdp_accountant_get_epsilon_with_diagnostics() -> None:
    accountant = RDPAccountant(delta=1e-5)
    accountant.step(sigma=1.0, clipping_norm=1.0, num_steps=10)
    eps, best_alpha = accountant.get_epsilon_with_diagnostics()
    assert eps > 0
    assert 2.0 <= best_alpha <= 100.0


def test_diagnostics_skips_invalid_epsilons() -> None:
    accountant = RDPAccountant(delta=1e-5)
    accountant._rdp_per_alpha = np.where(
        np.arange(len(RDP_ALPHAS)) % 2 == 0,
        -float("inf"),
        np.full_like(RDP_ALPHAS, 1.0),
    )
    accountant._steps.append({"sigma": 1.0, "clipping_norm": 1.0, "num_steps": 1})
    eps, best_alpha = accountant.get_epsilon_with_diagnostics()
    assert np.isfinite(eps)
    assert best_alpha >= 2.0


def test_rdp_accountant_diagnostics_empty() -> None:
    accountant = RDPAccountant(delta=1e-5)
    eps, best_alpha = accountant.get_epsilon_with_diagnostics()
    assert eps == 0.0
    assert best_alpha == 0.0


class TestEnforceEpsilonBudget:
    def test_valid_epsilon_unchanged(self) -> None:
        accountant = RDPAccountant(delta=1e-5)
        result_eps, result_sigma = enforce_epsilon_budget(
            candidate_epsilon=2.0,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=8.0,
            epsilon_min=0.1,
            clipping_norm=1.0,
            delta=1e-5,
        )
        assert result_eps == pytest.approx(2.0, rel=1e-3)
        assert result_sigma > 0

    def test_reduces_epsilon_when_budget_violated(self) -> None:
        accountant = RDPAccountant(delta=1e-5)
        accountant.step(sigma=10.0, clipping_norm=1.0, num_steps=1)
        result_eps, result_sigma = enforce_epsilon_budget(
            candidate_epsilon=5.0,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=2.0,
            epsilon_min=0.1,
            clipping_norm=1.0,
            delta=1e-5,
        )
        assert 0.1 <= result_eps < 5.0
        assert result_sigma > 0

    def test_exhausted_budget_returns_sentinel(self) -> None:
        accountant = RDPAccountant(delta=1e-5)
        accountant.step(sigma=0.1, clipping_norm=1.0, num_steps=200)
        result_eps, result_sigma = enforce_epsilon_budget(
            candidate_epsilon=1.0,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=0.5,
            epsilon_min=0.5,
            clipping_norm=1.0,
            delta=1e-5,
        )
        assert result_eps == pytest.approx(-1.0)
        assert result_sigma == pytest.approx(0.0)

    def test_candidate_below_min_returns_candidate(self) -> None:
        accountant = RDPAccountant(delta=1e-5)
        result_eps, result_sigma = enforce_epsilon_budget(
            candidate_epsilon=0.05,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=8.0,
            epsilon_min=0.1,
            clipping_norm=1.0,
            delta=1e-5,
        )
        assert result_eps == pytest.approx(0.05)
        assert result_sigma > 0

    def test_exhaustion_with_binary_search(self) -> None:
        accountant = RDPAccountant(delta=1e-5)
        accountant.step(sigma=0.3, clipping_norm=1.0, num_steps=100)
        result_eps, result_sigma = enforce_epsilon_budget(
            candidate_epsilon=10.0,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=1.0,
            epsilon_min=0.1,
            clipping_norm=1.0,
            delta=1e-5,
        )
        assert result_eps == pytest.approx(-1.0)
        assert result_sigma == pytest.approx(0.0)

    def test_returned_sigma_matches_epsilon(self) -> None:
        """Verify that the returned sigma, when used to calibrate,
        produces an epsilon close to the returned epsilon."""
        from src.privacy.per_update_dp import _rdp_epsilon_for_sigma
        accountant = RDPAccountant(delta=1e-5)
        result_eps, result_sigma = enforce_epsilon_budget(
            candidate_epsilon=2.0,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=8.0,
            epsilon_min=0.1,
            clipping_norm=1.0,
            delta=1e-5,
        )
        eps_from_sigma = _rdp_epsilon_for_sigma(result_sigma, 1.0, 1e-5)
        assert result_eps == pytest.approx(eps_from_sigma, rel=1e-6)


# ---------------------------------------------------------------------------
# DP-SGD RDP cost and sigma calibration
# ---------------------------------------------------------------------------


class TestComputeRdpCostDpSgd:
    def test_basic(self) -> None:
        cost = compute_rdp_cost_dp_sgd(alpha=2.0, sigma=1.0, sampling_rate=0.1)
        # alpha * q^2 / (2 * sigma^2) = 2 * 0.01 / 2 = 0.01
        assert cost == pytest.approx(0.01)

    def test_doubles_sigma_quarters_cost(self) -> None:
        c1 = compute_rdp_cost_dp_sgd(alpha=2.0, sigma=1.0, sampling_rate=0.1)
        c2 = compute_rdp_cost_dp_sgd(alpha=2.0, sigma=2.0, sampling_rate=0.1)
        assert c2 == pytest.approx(c1 / 4)

    def test_invalid_sigma(self) -> None:
        with pytest.raises(ValueError, match="sigma must be positive"):
            compute_rdp_cost_dp_sgd(alpha=2.0, sigma=0.0, sampling_rate=0.1)

    def test_invalid_sampling_rate_zero(self) -> None:
        with pytest.raises(ValueError, match="sampling_rate must be in"):
            compute_rdp_cost_dp_sgd(alpha=2.0, sigma=1.0, sampling_rate=0.0)

    def test_invalid_sampling_rate_above_one(self) -> None:
        with pytest.raises(ValueError, match="sampling_rate must be in"):
            compute_rdp_cost_dp_sgd(alpha=2.0, sigma=1.0, sampling_rate=1.5)


class TestCalibrateSigmaDpSgd:
    def test_basic(self) -> None:
        sigma = calibrate_sigma_dp_sgd(epsilon=1.0, sampling_rate=0.1, delta=1e-5)
        assert sigma > 0
        assert np.isfinite(sigma)

    def test_larger_epsilon_gives_smaller_sigma(self) -> None:
        s1 = calibrate_sigma_dp_sgd(epsilon=0.5, sampling_rate=0.1, delta=1e-5)
        s2 = calibrate_sigma_dp_sgd(epsilon=2.0, sampling_rate=0.1, delta=1e-5)
        assert s1 > s2

    def test_invalid_epsilon(self) -> None:
        with pytest.raises(ValueError, match="epsilon must be positive"):
            calibrate_sigma_dp_sgd(epsilon=0.0, sampling_rate=0.1, delta=1e-5)

    def test_invalid_sampling_rate(self) -> None:
        with pytest.raises(ValueError, match="sampling_rate must be in"):
            calibrate_sigma_dp_sgd(epsilon=1.0, sampling_rate=0.0, delta=1e-5)


class TestRdpAccountantPerExampleMode:
    def test_step_per_example(self) -> None:
        accountant = RDPAccountant(delta=1e-5)
        accountant.step(sigma=1.0, clipping_norm=0.1, num_steps=10, mode="per_example")
        eps = accountant.get_epsilon()
        assert eps > 0

    def test_per_example_costs_less_than_per_update(self) -> None:
        """Per-example with small sampling rate should cost less than per-update."""
        acc_update = RDPAccountant(delta=1e-5)
        acc_update.step(sigma=1.0, clipping_norm=1.0, num_steps=1, mode="per_update")

        acc_example = RDPAccountant(delta=1e-5)
        acc_example.step(sigma=1.0, clipping_norm=0.01, num_steps=1, mode="per_example")

        assert acc_example.get_epsilon() < acc_update.get_epsilon()

    def test_state_roundtrip_preserves_mode(self) -> None:
        accountant = RDPAccountant(delta=1e-5)
        accountant.step(sigma=1.0, clipping_norm=0.1, num_steps=5, mode="per_example")
        state = accountant.get_state()
        restored = RDPAccountant.from_state(state)
        assert restored.get_epsilon() == pytest.approx(accountant.get_epsilon())
        assert restored._steps[0]["mode"] == "per_example"


class TestEnforceBudgetPerExample:
    def test_per_example_mode(self) -> None:
        accountant = RDPAccountant(delta=1e-5)
        result_eps, result_sigma = enforce_epsilon_budget(
            candidate_epsilon=2.0,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=8.0,
            epsilon_min=0.1,
            clipping_norm=0.0,
            delta=1e-5,
            clipping_mode="per_example",
            sampling_rate=0.1,
        )
        assert result_eps == pytest.approx(2.0, rel=1e-3)
        assert result_sigma > 0

    def test_per_example_budget_violated(self) -> None:
        accountant = RDPAccountant(delta=1e-5)
        accountant.step(sigma=0.1, clipping_norm=0.1, num_steps=200, mode="per_example")
        result_eps, result_sigma = enforce_epsilon_budget(
            candidate_epsilon=1.0,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=0.5,
            epsilon_min=0.5,
            clipping_norm=0.0,
            delta=1e-5,
            clipping_mode="per_example",
            sampling_rate=0.1,
        )
        assert result_eps == pytest.approx(-1.0)
        assert result_sigma == pytest.approx(0.0)

    def test_per_example_num_steps_makes_budget_more_conservative(self) -> None:
        """Budget enforcement with num_steps > 1 should return a smaller (more
        conservative) epsilon or a larger sigma than without num_steps, because
        the actual per-round RDP cost is num_steps × per-step cost."""
        sampling_rate = 0.1
        budget = 10.0
        accountant = RDPAccountant(delta=1e-5)

        eps_no_steps, sigma_no_steps = enforce_epsilon_budget(
            candidate_epsilon=5.0,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=budget,
            epsilon_min=0.1,
            clipping_norm=0.0,
            delta=1e-5,
            clipping_mode="per_example",
            num_steps=1,
            sampling_rate=sampling_rate,
        )

        eps_with_steps, sigma_with_steps = enforce_epsilon_budget(
            candidate_epsilon=5.0,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=budget,
            epsilon_min=0.1,
            clipping_norm=0.0,
            delta=1e-5,
            clipping_mode="per_example",
            num_steps=10,
            sampling_rate=sampling_rate,
        )

        # With num_steps=10, the enforcement sees 10× the cost, so it must
        # either reduce epsilon or increase sigma (more noise).
        assert sigma_with_steps >= sigma_no_steps
        if eps_no_steps > 0 and eps_with_steps > 0:
            assert eps_with_steps <= eps_no_steps

    def test_per_example_num_steps_exhausts_budget_faster(self) -> None:
        """After accumulating actual steps (accountant), enforce with num_steps
        should detect budget exhaustion earlier than without num_steps."""
        sampling_rate = 0.1
        accountant = RDPAccountant(delta=1e-5)
        # Accumulate 5 rounds × 10 steps = 50 actual steps
        for _ in range(5):
            accountant.step(sigma=0.5, clipping_norm=sampling_rate,
                            num_steps=10, mode="per_example")

        eps_no_steps, _ = enforce_epsilon_budget(
            candidate_epsilon=1.0,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=100.0,
            epsilon_min=0.1,
            clipping_norm=0.0,
            delta=1e-5,
            clipping_mode="per_example",
            num_steps=1,
            sampling_rate=sampling_rate,
        )

        eps_with_steps, _ = enforce_epsilon_budget(
            candidate_epsilon=1.0,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=100.0,
            epsilon_min=0.1,
            clipping_norm=0.0,
            delta=1e-5,
            clipping_mode="per_example",
            num_steps=10,
            sampling_rate=sampling_rate,
        )

        # Both may succeed with a large budget, but with_steps should be
        # more constrained (smaller or equal epsilon).
        if eps_no_steps > 0 and eps_with_steps > 0:
            assert eps_with_steps <= eps_no_steps


class TestResolveEpsilon:
    """Tests for _resolve_epsilon in client_app.py."""

    def _make_config(self, personalization_enabled: bool = True,
                     bo_eps_min: float = 0.1) -> ExperimentConfig:
        cfg = ExperimentConfig()
        cfg.privacy.enabled = True
        cfg.privacy.update_clip_norm = 1.0
        cfg.privacy.delta = 1e-5
        cfg.privacy.target_epsilon = 5.0
        cfg.personalization.enabled = personalization_enabled
        cfg.bo.epsilon_min = bo_eps_min
        return cfg

    def test_explicit_eps_min_respected_with_personalization(self) -> None:
        from src.client_app import _resolve_epsilon
        config = self._make_config()
        accountant = RDPAccountant(delta=1e-5)
        result_eps, result_sigma, _candidate, _bo, _acct = _resolve_epsilon(
            scheduler=None,
            accountant=accountant,
            config=config,
            total_budget=100.0,
            eps_min=0.1,
        )
        assert result_eps >= 0.1 or result_eps == -1.0
        assert result_sigma >= 0

    def test_lower_bound_above_personalization_unchanged(self) -> None:
        from src.client_app import _resolve_epsilon
        config = self._make_config()
        accountant = RDPAccountant(delta=1e-5)
        result_eps, result_sigma, _candidate, _bo, _acct = _resolve_epsilon(
            scheduler=None,
            accountant=accountant,
            config=config,
            total_budget=100.0,
            eps_min=2.0,
        )
        assert result_eps >= 2.0 or result_eps == -1.0
        assert result_sigma >= 0

    def test_no_personalization_uses_bo_min(self) -> None:
        from src.client_app import _resolve_epsilon
        config = self._make_config(personalization_enabled=False, bo_eps_min=0.3)
        accountant = RDPAccountant(delta=1e-5)
        result_eps, result_sigma, _candidate, _bo, _acct = _resolve_epsilon(
            scheduler=None,
            accountant=accountant,
            config=config,
            total_budget=100.0,
            eps_min=None,
        )
        assert result_eps >= 0.3 or result_eps == -1.0
        assert result_sigma >= 0


# ---------------------------------------------------------------------------
# compute_utility_loss — logit clamping prevents NaN
# ---------------------------------------------------------------------------


class TestComputeUtilityLoss:
    def test_extreme_logits_produce_finite_loss(self) -> None:
        model = nn.Linear(10, 2)
        with torch.no_grad():
            model.weight.fill_(100.0)
            model.bias.fill_(0.0)
        data = TensorDataset(torch.randn(4, 10), torch.randint(0, 2, (4,)))
        loader = DataLoader(data, batch_size=4)
        loss = compute_utility_loss(model, loader)
        assert math.isfinite(loss), f"Loss should be finite, got {loss}"

    def test_normal_logits_unchanged(self) -> None:
        model = nn.Linear(10, 2)
        with torch.no_grad():
            model.weight.fill_(0.1)
            model.bias.fill_(0.0)
        data = TensorDataset(torch.randn(4, 10), torch.randint(0, 2, (4,)))
        loader = DataLoader(data, batch_size=4)
        loss = compute_utility_loss(model, loader)
        assert math.isfinite(loss)


# ---------------------------------------------------------------------------
# Weight clamping — noise is bounded after mechanism apply
# ---------------------------------------------------------------------------


class TestMomentumAccountingUnchanged:
    """Momentum is applied pre-noise, so sigma and the RDP cost are unchanged."""

    def test_sigma_and_rdp_cost_identical_with_and_without_momentum(self) -> None:
        from src.client.per_example_dp_client import PerExampleDPClient
        from src.models.base import BaseModel

        class _SimpleModel(BaseModel):
            def __init__(self) -> None:
                self._net = nn.Linear(10, 2)

            def get_model(self) -> nn.Module:
                return self._net

        def make_loader() -> DataLoader[Any]:
            data = TensorDataset(torch.randn(4, 10), torch.randint(0, 2, (4,)))
            return DataLoader(data, batch_size=2)

        def make_config() -> ExperimentConfig:
            config = ExperimentConfig()
            config.privacy.enabled = True
            config.privacy.clipping_mode = "per_example"
            config.privacy.accountant_mode = "rdp_native"
            config.data.batch_size = 2
            return config

        loader = make_loader()
        params = None

        clients = []
        for momentum in (0.0, 0.9):
            config = make_config()
            config.optimizer.momentum = momentum
            model = _SimpleModel()
            client = PerExampleDPClient(
                model, loader, loader, config,
                client_epsilon=0.5, seed=42,
            )
            if params is None:
                params = client.get_parameters({})
            clients.append((client, config, model))

        metrics = []
        for client, _config, _model in clients:
            _, _num_examples, m = client.fit(params or [], {})
            metrics.append(m)

        assert metrics[0]["sigma"] == pytest.approx(metrics[1]["sigma"])
        assert metrics[0]["rdp_cost"] == pytest.approx(metrics[1]["rdp_cost"])



