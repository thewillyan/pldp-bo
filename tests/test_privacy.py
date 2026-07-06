from __future__ import annotations

import numpy as np
import pytest

from src.privacy import RDPAccountant, simulate_epsilon
from src.privacy.per_update_dp import (
    PerUpdateGaussianMechanism,
    add_gaussian_noise,
    calibrate_sigma,
    clip_update,
    compute_rdp_cost,
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


def test_rdp_accountant_get_state() -> None:
    accountant = RDPAccountant(delta=1e-5)
    for _ in range(3):
        accountant.step(sigma=0.5, clipping_norm=1.0)

    state = accountant.get_state()
    assert "delta" in state
    assert "steps" in state
    assert len(state["steps"]) == 3
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
    state = {"delta": 1e-6, "steps": [], "rdp_per_alpha": []}
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
