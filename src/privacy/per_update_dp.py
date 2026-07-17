from __future__ import annotations

import logging
import math

import numpy as np

from src.privacy.constants import RDP_ALPHAS
from src.utils import deserialize_rng, serialize_rng

logger = logging.getLogger(__name__)


def _rdp_epsilon_for_sigma(
    sigma: float,
    clipping_norm: float,
    delta: float,
) -> float:
    cost = np.array(
        [compute_rdp_cost(float(a), sigma, clipping_norm) for a in RDP_ALPHAS],
        dtype=np.float64,
    )
    log_one_over_delta = math.log(1.0 / delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        epsilons = cost + log_one_over_delta / (RDP_ALPHAS - 1.0)
    valid = np.isfinite(epsilons)
    if not valid.any():
        return float("inf")
    return float(np.min(epsilons[valid]))


def _rdp_calibrate_sigma(
    epsilon: float,
    clipping_norm: float,
    delta: float,
    sigma_min: float = 0.01,
    sigma_max: float = 5e7,
) -> float:
    eps_at_max = _rdp_epsilon_for_sigma(sigma_max, clipping_norm, delta)
    if eps_at_max > epsilon:
        logger.warning(
            "_rdp_calibrate_sigma: target epsilon=%.4f is below the "
            "fundamental RDP lower bound ≈%.4f for delta=%.0e. "
            "Returning sigma_max=%.0e; actual privacy will be stronger "
            "than requested but epsilon will be clamped to ≈%.4f.",
            epsilon, eps_at_max, delta, sigma_max, eps_at_max,
        )
        return sigma_max
    eps_at_min = _rdp_epsilon_for_sigma(sigma_min, clipping_norm, delta)
    if eps_at_min < epsilon:
        return sigma_min
    lo, hi = sigma_min, sigma_max
    for _ in range(50):
        mid = (lo + hi) / 2.0
        eps = _rdp_epsilon_for_sigma(mid, clipping_norm, delta)
        if eps > epsilon:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def calibrate_sigma(
    epsilon: float,
    clipping_norm: float,
    delta: float,
    min_sigma: float | None = None,
) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    if clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive")
    sigma = _rdp_calibrate_sigma(epsilon, clipping_norm, delta)
    floor = min_sigma if min_sigma is not None else clipping_norm
    if sigma < floor:
        logger.warning(
            "calibrate_sigma: RDP-calibrated sigma=%.6f for clipping_norm=%.2f, "
            "epsilon=%.2f is below floor %.6f; clamping to floor. "
            "Actual privacy will be stronger than requested epsilon.",
            sigma, clipping_norm, epsilon, floor,
        )
        return floor
    return sigma


def _clip_update(delta: np.ndarray, clipping_norm: float) -> np.ndarray:
    norm = np.linalg.norm(delta)
    if norm > clipping_norm:
        return delta * (clipping_norm / norm)
    return delta


def add_gaussian_noise(
    clipped_delta: np.ndarray,
    sigma: float,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState()
    noise = rng.normal(0, sigma, size=clipped_delta.shape).astype(clipped_delta.dtype)
    return clipped_delta + noise


def compute_rdp_cost(alpha: float, sigma: float, clipping_norm: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive")
    return alpha * clipping_norm**2 / (2.0 * sigma**2)


class PerUpdateGaussianMechanism:
    def __init__(self, clipping_norm: float, delta: float, seed: int | None = None) -> None:
        if clipping_norm <= 0:
            raise ValueError("clipping_norm must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        self._clipping_norm = clipping_norm
        self._delta = delta
        self._rng = np.random.RandomState(seed)

    def apply(self, delta: np.ndarray, epsilon: float, sigma: float | None = None) -> tuple[np.ndarray, float]:
        if sigma is None:
            sigma = calibrate_sigma(epsilon, self._clipping_norm, self._delta)
        clipped = _clip_update(delta, self._clipping_norm)
        noisy = add_gaussian_noise(clipped, sigma, rng=self._rng)
        return noisy, sigma

    def get_state(self) -> dict:
        return {"rng_state": serialize_rng(self._rng)}

    @classmethod
    def from_state(
        cls, state: dict, clipping_norm: float, delta: float,
    ) -> PerUpdateGaussianMechanism:
        mechanism = cls(clipping_norm=clipping_norm, delta=delta)
        mechanism._rng.set_state(deserialize_rng(state["rng_state"]))
        return mechanism

    @property
    def clipping_norm(self) -> float:
        return self._clipping_norm

    @property
    def delta(self) -> float:
        return self._delta



def _hypothetical_epsilon(
    current_rdp: np.ndarray,
    sigma: float,
    clipping_norm: float,
    delta: float,
) -> float:
    cost = np.array(
        [compute_rdp_cost(float(a), sigma, clipping_norm) for a in RDP_ALPHAS],
        dtype=np.float64,
    )
    total_rdp = current_rdp + cost
    log_one_over_delta = math.log(1.0 / delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        epsilons = total_rdp + log_one_over_delta / (RDP_ALPHAS - 1.0)
    valid = np.isfinite(epsilons)
    if not valid.any():
        return float("inf")
    return float(np.min(epsilons[valid]))


def _is_epsilon_within_budget(
    epsilon: float,
    current_rdp: np.ndarray,
    epsilon_budget: float,
    clipping_norm: float,
    delta: float,
) -> bool:
    sigma = calibrate_sigma(epsilon, clipping_norm, delta)
    projected = _hypothetical_epsilon(current_rdp, sigma, clipping_norm, delta)
    return projected <= epsilon_budget


def enforce_epsilon_budget(
    candidate_epsilon: float,
    current_rdp: np.ndarray,
    epsilon_budget: float,
    epsilon_min: float,
    clipping_norm: float,
    delta: float,
) -> tuple[float, float]:
    """Return (ε, σ) where ε is the largest epsilon ≤ candidate_epsilon that fits the budget,
    and σ is the corresponding noise scale.

    Binary-searches over σ directly instead of ε to avoid calling
    ``calibrate_sigma`` (itself a binary search) inside the outer loop.

    Returns (-1.0, 0.0) if even *epsilon_min* would exceed the remaining budget,
    signalling that the client's privacy budget is exhausted.
    """
    if candidate_epsilon <= 0:
        return -1.0, 0.0

    if candidate_epsilon <= epsilon_min:
        if _is_epsilon_within_budget(candidate_epsilon, current_rdp,
                                     epsilon_budget, clipping_norm, delta):
            sigma = calibrate_sigma(candidate_epsilon, clipping_norm, delta)
            return candidate_epsilon, sigma
        return -1.0, 0.0

    sigma_candidate = calibrate_sigma(candidate_epsilon, clipping_norm, delta)
    hypothetical = _hypothetical_epsilon(
        current_rdp, sigma_candidate, clipping_norm, delta,
    )

    if hypothetical <= epsilon_budget:
        return candidate_epsilon, sigma_candidate

    if not _is_epsilon_within_budget(epsilon_min, current_rdp, epsilon_budget,
                                     clipping_norm, delta):
        return -1.0, 0.0

    # Binary search over σ (monotonic: larger σ → smaller ε → lower RDP cost).
    sigma_min_eps = calibrate_sigma(epsilon_min, clipping_norm, delta)
    lo_sigma, hi_sigma = sigma_candidate, sigma_min_eps

    for _ in range(30):
        mid_sigma = (lo_sigma + hi_sigma) / 2.0
        projected = _hypothetical_epsilon(
            current_rdp, mid_sigma, clipping_norm, delta,
        )
        if projected <= epsilon_budget:
            hi_sigma = mid_sigma  # this σ fits; try smaller σ (larger ε)
        else:
            lo_sigma = mid_sigma  # too expensive; need larger σ

    result_sigma = (lo_sigma + hi_sigma) / 2.0
    result_epsilon = _rdp_epsilon_for_sigma(result_sigma, clipping_norm, delta)
    return result_epsilon, result_sigma
