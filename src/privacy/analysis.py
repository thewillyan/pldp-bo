from __future__ import annotations

import numpy as np

from src.privacy.accountant import RDPAccountant


def simulate_epsilon(
    num_rounds: int,
    noise_multiplier: float,
    sampling_rate: float,
    local_steps: int = 1,
    delta: float = 1e-5,
) -> list[float]:
    accountant = RDPAccountant(delta=delta)
    epsilons: list[float] = []
    for r in range(num_rounds):
        accountant.step(
            noise_multiplier=noise_multiplier,
            sample_rate=sampling_rate,
            num_steps=local_steps,
        )
        epsilons.append(accountant.get_epsilon())
    return epsilons


def find_noise_for_target_epsilon(
    target_epsilon: float,
    num_rounds: int,
    sampling_rate: float,
    local_steps: int = 1,
    delta: float = 1e-5,
) -> float:
    from src.privacy.dp_mechanism import calibrate_gaussian_noise

    return calibrate_gaussian_noise(
        target_epsilon=target_epsilon,
        delta=delta,
        sampling_rate=sampling_rate,
        steps=num_rounds * local_steps,
    )
