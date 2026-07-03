from __future__ import annotations

import pytest

from src.privacy import RDPAccountant, calibrate_gaussian_noise, simulate_epsilon


def test_rdp_accountant_initial() -> None:
    accountant = RDPAccountant(delta=1e-5)
    eps = accountant.get_epsilon()
    assert eps == 0.0


def test_rdp_accountant_steps() -> None:
    accountant = RDPAccountant(delta=1e-5)
    for _ in range(10):
        accountant.step(noise_multiplier=1.0, sample_rate=0.01)
    eps = accountant.get_epsilon()
    assert eps > 0


def test_rdp_accountant_privacy_spent() -> None:
    accountant = RDPAccountant(delta=1e-5)
    for _ in range(5):
        accountant.step(noise_multiplier=1.0, sample_rate=0.01)
    spent = accountant.get_privacy_spent()
    assert "epsilon" in spent
    assert "delta" in spent
    assert spent["delta"] == 1e-5


def test_simulate_epsilon() -> None:
    epsilons = simulate_epsilon(
        num_rounds=10,
        noise_multiplier=1.0,
        sampling_rate=0.01,
    )
    assert len(epsilons) == 10
    assert epsilons[-1] > epsilons[0]


def test_calibrate_gaussian_noise() -> None:
    sigma = calibrate_gaussian_noise(
        target_epsilon=1.0,
        delta=1e-5,
        sampling_rate=0.01,
        steps=100,
    )
    assert sigma > 0
