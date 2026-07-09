from __future__ import annotations

import numpy as np
import pytest

from unittest.mock import MagicMock

from src.config.loader import ExperimentConfig
from src.privacy import RDPAccountant, simulate_epsilon
from src.privacy.constants import RDP_ALPHAS
from src.privacy.per_update_dp import (
    PerUpdateGaussianMechanism,
    add_gaussian_noise,
    calibrate_sigma,
    clip_update,
    compute_rdp_cost,
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


def test_calibrate_sigma_minimum_is_clipping_norm() -> None:
    sigma = calibrate_sigma(epsilon=100.0, clipping_norm=1.0, delta=1e-5)
    assert sigma >= 1.0
    assert sigma == pytest.approx(1.0)
    sigma_large = calibrate_sigma(epsilon=1e9, clipping_norm=5.0, delta=1e-5)
    assert sigma_large == pytest.approx(5.0)


def test_calibrate_sigma_invalid_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon must be positive"):
        calibrate_sigma(epsilon=0.0, clipping_norm=1.0, delta=1e-5)


def test_clip_update_below_threshold() -> None:
    delta = np.array([0.1, 0.2, 0.3])
    clipped = clip_update(delta, clipping_norm=1.0)
    np.testing.assert_array_almost_equal(clipped, delta)


def test_clip_update_above_threshold() -> None:
    delta = np.array([3.0, 4.0])  # norm = 5.0
    clipped = clip_update(delta, clipping_norm=1.0)
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
    clipped_norm = np.linalg.norm(clip_update(delta, 1.0))
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
        result = enforce_epsilon_budget(
            candidate_epsilon=2.0,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=8.0,
            epsilon_min=0.1,
            clipping_norm=1.0,
            delta=1e-5,
        )
        assert result == pytest.approx(2.0, rel=1e-3)

    def test_reduces_epsilon_when_budget_violated(self) -> None:
        accountant = RDPAccountant(delta=1e-5)
        accountant.step(sigma=10.0, clipping_norm=1.0, num_steps=1)
        result = enforce_epsilon_budget(
            candidate_epsilon=5.0,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=2.0,
            epsilon_min=0.1,
            clipping_norm=1.0,
            delta=1e-5,
        )
        assert 0.1 <= result < 5.0

    def test_exhausted_budget_returns_sentinel(self) -> None:
        accountant = RDPAccountant(delta=1e-5)
        accountant.step(sigma=0.1, clipping_norm=1.0, num_steps=200)
        result = enforce_epsilon_budget(
            candidate_epsilon=1.0,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=0.5,
            epsilon_min=0.5,
            clipping_norm=1.0,
            delta=1e-5,
        )
        assert result == pytest.approx(-1.0)

    def test_candidate_below_min_returns_candidate(self) -> None:
        accountant = RDPAccountant(delta=1e-5)
        result = enforce_epsilon_budget(
            candidate_epsilon=0.05,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=8.0,
            epsilon_min=0.1,
            clipping_norm=1.0,
            delta=1e-5,
        )
        assert result == pytest.approx(0.05)

    def test_exhaustion_with_binary_search(self) -> None:
        accountant = RDPAccountant(delta=1e-5)
        accountant.step(sigma=0.3, clipping_norm=1.0, num_steps=100)
        result = enforce_epsilon_budget(
            candidate_epsilon=10.0,
            current_rdp=accountant.rdp_per_alpha,
            epsilon_budget=1.0,
            epsilon_min=0.1,
            clipping_norm=1.0,
            delta=1e-5,
        )
        assert result == pytest.approx(-1.0)


class TestResolveEpsilon:
    """Tests for _resolve_epsilon in client_app.py."""

    def _make_config(self, personalization_enabled: bool = True,
                     personalization_eps_min: float = 1.0,
                     bo_eps_min: float = 0.1) -> ExperimentConfig:
        cfg = ExperimentConfig()
        cfg.privacy.enabled = True
        cfg.privacy.update_clip_norm = 1.0
        cfg.privacy.delta = 1e-5
        cfg.privacy.target_epsilon = 5.0
        cfg.personalization.enabled = personalization_enabled
        cfg.personalization.epsilon_min = personalization_eps_min
        cfg.bo.epsilon_min = bo_eps_min
        return cfg

    def test_explicit_eps_min_respected_with_personalization(self) -> None:
        from src.client_app import _resolve_epsilon
        config = self._make_config(personalization_eps_min=1.0)
        accountant = RDPAccountant(delta=1e-5)
        result = _resolve_epsilon(
            scheduler=None,
            accountant=accountant,
            config=config,
            total_budget=100.0,
            eps_min=0.1,
        )
        assert result >= 1.0 or result == -1.0

    def test_lower_bound_above_personalization_unchanged(self) -> None:
        from src.client_app import _resolve_epsilon
        config = self._make_config(personalization_eps_min=0.5)
        accountant = RDPAccountant(delta=1e-5)
        result = _resolve_epsilon(
            scheduler=None,
            accountant=accountant,
            config=config,
            total_budget=100.0,
            eps_min=2.0,
        )
        assert result >= 2.0 or result == -1.0

    def test_no_personalization_uses_bo_min(self) -> None:
        from src.client_app import _resolve_epsilon
        config = self._make_config(personalization_enabled=False, bo_eps_min=0.3)
        accountant = RDPAccountant(delta=1e-5)
        result = _resolve_epsilon(
            scheduler=None,
            accountant=accountant,
            config=config,
            total_budget=100.0,
            eps_min=None,
        )
        assert result >= 0.3 or result == -1.0
