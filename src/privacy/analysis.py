from __future__ import annotations

import math

import numpy as np

from src.privacy.accountant import RDPAccountant
from src.privacy.constants import RDP_ALPHAS
from src.privacy.per_update_dp import compute_rdp_cost


def simulate_epsilon(
    num_rounds: int,
    sigma: float,
    clipping_norm: float = 1.0,
    delta: float = 1e-5,
) -> list[float]:
    cost_per_alpha = np.array(
        [compute_rdp_cost(float(a), sigma, clipping_norm) for a in RDP_ALPHAS],
        dtype=np.float64,
    )
    log_one_over_delta = math.log(1.0 / delta)
    epsilons: list[float] = []
    cumulative = np.zeros_like(cost_per_alpha)
    for _ in range(num_rounds):
        cumulative += cost_per_alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            eps_vals = cumulative + log_one_over_delta / (RDP_ALPHAS - 1.0)
        valid = np.isfinite(eps_vals)
        if not valid.any():
            epsilons.append(float("inf"))
        else:
            epsilons.append(float(np.min(eps_vals[valid])))
    return epsilons


def find_noise_for_target_epsilon(
    target_epsilon: float,
    num_rounds: int,
    clipping_norm: float = 1.0,
    delta: float = 1e-5,
    sigma_bounds: tuple[float, float] = (0.1, 100.0),
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
        acc.step(sigma=sigma, clipping_norm=clipping_norm, num_steps=num_rounds)
        return acc.get_epsilon()

    lo, hi = sigma_bounds

    eps_at_hi = _compute_eps(hi)
    if eps_at_hi > target_epsilon:
        raise ValueError(
            f"Cannot achieve target epsilon {target_epsilon} "
            f"with sigma ≤ {hi}. eps({hi}) = {eps_at_hi:.4f}. "
            "Increase max sigma or reduce target epsilon."
        )

    eps_at_lo = _compute_eps(lo)
    if eps_at_lo < target_epsilon:
        return lo

    for _ in range(50):
        mid = (lo + hi) / 2.0
        eps = _compute_eps(mid)
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
