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


def test_rdp_accountant_get_state() -> None:
    accountant = RDPAccountant(delta=1e-5)
    for _ in range(3):
        accountant.step(noise_multiplier=0.5, sample_rate=0.02)

    state = accountant.get_state()
    assert "delta" in state
    assert "steps" in state
    assert len(state["steps"]) == 3
    assert state["delta"] == 1e-5


def test_rdp_accountant_from_state_roundtrip() -> None:
    original = RDPAccountant(delta=1e-5)
    for _ in range(5):
        original.step(noise_multiplier=1.0, sample_rate=0.01)

    state = original.get_state()
    restored = RDPAccountant.from_state(state)

    assert restored.get_epsilon() == pytest.approx(original.get_epsilon())
    assert restored.total_steps() == original.total_steps()
    assert restored._delta == original._delta


def test_rdp_accountant_from_state_empty() -> None:
    state = {"delta": 1e-6, "steps": []}
    accountant = RDPAccountant.from_state(state)

    assert accountant.get_epsilon() == 0.0
    assert accountant.total_steps() == 0
    assert accountant._delta == 1e-6


def test_rdp_accountant_reset() -> None:
    accountant = RDPAccountant(delta=1e-5)
    for _ in range(5):
        accountant.step(noise_multiplier=1.0, sample_rate=0.01)

    assert accountant.get_epsilon() > 0
    accountant.reset()
    assert accountant.get_epsilon() == 0.0
    assert accountant.total_steps() == 0
