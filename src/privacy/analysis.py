from __future__ import annotations

from src.privacy.accountant import RDPAccountant


def simulate_epsilon(
    num_rounds: int,
    sigma: float,
    clipping_norm: float = 1.0,
    delta: float = 1e-5,
) -> list[float]:
    accountant = RDPAccountant(delta=delta)
    epsilons: list[float] = []
    for _ in range(num_rounds):
        accountant.step(sigma=sigma, clipping_norm=clipping_norm, num_steps=1)
        epsilons.append(accountant.get_epsilon())
    return epsilons


def find_noise_for_target_epsilon(
    target_epsilon: float,
    num_rounds: int,
    clipping_norm: float = 1.0,
    delta: float = 1e-5,
) -> float:
    if target_epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    if num_rounds < 1:
        raise ValueError("num_rounds must be positive")
    if clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive")

    def _compute_eps(sigma: float) -> float:
        acc = RDPAccountant(delta=delta)
        for _ in range(num_rounds):
            acc.step(sigma=sigma, clipping_norm=clipping_norm, num_steps=1)
        return acc.get_epsilon()

    lo, hi = 0.1, 100.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        eps = _compute_eps(mid)
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
