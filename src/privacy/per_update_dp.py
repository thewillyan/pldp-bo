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
            epsilon,
            eps_at_max,
            delta,
            sigma_max,
            eps_at_max,
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
    if min_sigma is not None and sigma < min_sigma:
        logger.warning(
            "calibrate_sigma: RDP-calibrated sigma=%.6f for clipping_norm=%.2f, "
            "epsilon=%.2f is below min_sigma %.6f; clamping to min_sigma. "
            "Actual privacy will be stronger than requested epsilon.",
            sigma,
            clipping_norm,
            epsilon,
            min_sigma,
        )
        return min_sigma
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


def compute_rdp_cost_dp_sgd(alpha: float, sigma: float, sampling_rate: float) -> float:
    """Per-step RDP cost for DP-SGD with Poisson subsampling.

    Parameters
    ----------
    alpha : float
        RDP order.
    sigma : float
        Noise multiplier (actual noise std = sigma * C).
    sampling_rate : float
        Batch size / dataset size, i.e. probability of including any given example.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if sampling_rate <= 0 or sampling_rate > 1:
        raise ValueError("sampling_rate must be in (0, 1]")
    return alpha * sampling_rate**2 / (2.0 * sigma**2)


def _rdp_epsilon_for_sigma_dp_sgd(
    sigma: float,
    sampling_rate: float,
    delta: float,
) -> float:
    cost = np.array(
        [compute_rdp_cost_dp_sgd(float(a), sigma, sampling_rate) for a in RDP_ALPHAS],
        dtype=np.float64,
    )
    log_one_over_delta = math.log(1.0 / delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        epsilons = cost + log_one_over_delta / (RDP_ALPHAS - 1.0)
    valid = np.isfinite(epsilons)
    if not valid.any():
        return float("inf")
    return float(np.min(epsilons[valid]))


def _rdp_calibrate_sigma_dp_sgd(
    epsilon: float,
    sampling_rate: float,
    delta: float,
    sigma_min: float = 0.01,
    sigma_max: float = 5e7,
) -> float:
    eps_at_max = _rdp_epsilon_for_sigma_dp_sgd(sigma_max, sampling_rate, delta)
    if eps_at_max > epsilon:
        logger.warning(
            "_rdp_calibrate_sigma_dp_sgd: target epsilon=%.4f is below the "
            "fundamental RDP lower bound ≈%.4f for delta=%.0e. "
            "Returning sigma_max=%.0e; actual privacy will be stronger "
            "than requested but epsilon will be clamped to ≈%.4f.",
            epsilon,
            eps_at_max,
            delta,
            sigma_max,
            eps_at_max,
        )
        return sigma_max
    eps_at_min = _rdp_epsilon_for_sigma_dp_sgd(sigma_min, sampling_rate, delta)
    if eps_at_min < epsilon:
        return sigma_min
    lo, hi = sigma_min, sigma_max
    for _ in range(50):
        mid = (lo + hi) / 2.0
        eps = _rdp_epsilon_for_sigma_dp_sgd(mid, sampling_rate, delta)
        if eps > epsilon:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def calibrate_sigma_dp_sgd(
    epsilon: float,
    sampling_rate: float,
    delta: float,
    min_sigma: float | None = None,
) -> float:
    """Calibrate noise multiplier for DP-SGD given target epsilon and sampling rate."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    if sampling_rate <= 0 or sampling_rate > 1:
        raise ValueError("sampling_rate must be in (0, 1]")
    sigma = _rdp_calibrate_sigma_dp_sgd(epsilon, sampling_rate, delta)
    if min_sigma is not None and sigma < min_sigma:
        logger.warning(
            "calibrate_sigma_dp_sgd: RDP-calibrated sigma=%.6f for "
            "sampling_rate=%.4f, epsilon=%.2f is below min_sigma %.6f; "
            "clamping to min_sigma. Actual privacy will be stronger "
            "than requested epsilon.",
            sigma,
            sampling_rate,
            epsilon,
            min_sigma,
        )
        return min_sigma
    return sigma


def _sigma_for_rdp_target(
    rdp_target: float,
    alpha: float,
    clipping_norm: float,
) -> float:
    """Direct formula: sigma = sqrt(alpha * C^2 / (2 * rdp_target)).

    This computes the noise scale that yields exactly *rdp_target* RDP cost
    at the given Renyi order *alpha* for Gaussian mechanism with clipping
    norm *clipping_norm*.
    """
    if rdp_target <= 0:
        raise ValueError("rdp_target must be positive")
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")
    if clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive")
    return math.sqrt(alpha * clipping_norm**2 / (2.0 * rdp_target))


def _sigma_for_rdp_target_dp_sgd(
    rdp_target: float,
    alpha: float,
    sampling_rate: float,
) -> float:
    """Direct formula for DP-SGD with Poisson subsampling.

    sigma = sqrt(alpha * q^2 / (2 * rdp_target)).

    Here sigma is the noise multiplier (actual noise std = sigma * C).
    """
    if rdp_target <= 0:
        raise ValueError("rdp_target must be positive")
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")
    if sampling_rate <= 0 or sampling_rate > 1:
        raise ValueError("sampling_rate must be in (0, 1]")
    return math.sqrt(alpha * sampling_rate**2 / (2.0 * rdp_target))


def calibrate_sigma_rdp(
    rdp_target: float,
    alpha: float,
    clipping_norm: float,
    min_sigma: float | None = None,
) -> float:
    """Calibrate sigma from an RDP target at a fixed alpha — direct formula, no binary search."""
    if rdp_target <= 0:
        raise ValueError("rdp_target must be positive")
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")
    if clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive")
    sigma = _sigma_for_rdp_target(rdp_target, alpha, clipping_norm)
    if min_sigma is not None and sigma < min_sigma:
        logger.warning(
            "calibrate_sigma_rdp: sigma=%.6f for alpha=%.1f, rdp_target=%.6f "
            "is below min_sigma %.6f; clamping to min_sigma. "
            "Actual RDP cost will be lower than rdp_target.",
            sigma,
            alpha,
            rdp_target,
            min_sigma,
        )
        return min_sigma
    return sigma


def calibrate_sigma_rdp_dp_sgd(
    rdp_target: float,
    alpha: float,
    sampling_rate: float,
    min_sigma: float | None = None,
) -> float:
    """Calibrate noise multiplier for DP-SGD from RDP target at fixed alpha."""
    if rdp_target <= 0:
        raise ValueError("rdp_target must be positive")
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")
    if sampling_rate <= 0 or sampling_rate > 1:
        raise ValueError("sampling_rate must be in (0, 1]")
    sigma = _sigma_for_rdp_target_dp_sgd(rdp_target, alpha, sampling_rate)
    if min_sigma is not None and sigma < min_sigma:
        logger.warning(
            "calibrate_sigma_rdp_dp_sgd: sigma=%.6f for alpha=%.1f, "
            "rdp_target=%.6f is below min_sigma %.6f; clamping to min_sigma. "
            "Actual RDP cost will be lower than rdp_target.",
            sigma,
            alpha,
            rdp_target,
            min_sigma,
        )
        return min_sigma
    return sigma


class PerUpdateGaussianMechanism:
    def __init__(self, clipping_norm: float, delta: float, seed: int | None = None) -> None:
        if clipping_norm <= 0:
            raise ValueError("clipping_norm must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        self._clipping_norm = clipping_norm
        self._delta = delta
        self._rng = np.random.RandomState(seed)

    def apply(
        self,
        delta: np.ndarray,
        epsilon: float,
        sigma: float | None = None,
    ) -> tuple[np.ndarray, float]:
        if sigma is None:
            sigma = calibrate_sigma(epsilon, self._clipping_norm, self._delta)
        clipped = _clip_update(delta, self._clipping_norm)
        noisy = add_gaussian_noise(clipped, sigma, rng=self._rng)
        return noisy, sigma

    def get_state(self) -> dict:
        return {"rng_state": serialize_rng(self._rng)}

    @classmethod
    def from_state(
        cls,
        state: dict,
        clipping_norm: float,
        delta: float,
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
    clipping_mode: str = "per_update",
    num_steps: int = 1,
    sampling_rate: float | None = None,
) -> float:
    if clipping_mode == "per_example":
        rate = sampling_rate if sampling_rate is not None else clipping_norm
        cost = np.array(
            [compute_rdp_cost_dp_sgd(float(a), sigma, rate) for a in RDP_ALPHAS],
            dtype=np.float64,
        )
    else:
        cost = np.array(
            [compute_rdp_cost(float(a), sigma, clipping_norm) for a in RDP_ALPHAS],
            dtype=np.float64,
        )
    total_rdp = current_rdp + cost * num_steps
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
    clipping_mode: str = "per_update",
    num_steps: int = 1,
    sampling_rate: float | None = None,
) -> bool:
    if clipping_mode == "per_example":
        if sampling_rate is None:
            raise ValueError("sampling_rate is required for per_example mode")
        sigma = calibrate_sigma_dp_sgd(epsilon, sampling_rate, delta)
    else:
        sigma = calibrate_sigma(epsilon, clipping_norm, delta)
    projected = _hypothetical_epsilon(
        current_rdp,
        sigma,
        clipping_norm,
        delta,
        clipping_mode=clipping_mode,
        num_steps=num_steps,
        sampling_rate=sampling_rate,
    )
    return projected <= epsilon_budget


def enforce_epsilon_budget(
    candidate_epsilon: float,
    current_rdp: np.ndarray,
    epsilon_budget: float,
    epsilon_min: float,
    clipping_norm: float,
    delta: float,
    clipping_mode: str = "per_update",
    num_steps: int = 1,
    sampling_rate: float | None = None,
) -> tuple[float, float]:
    """Return (ε, σ) where ε is the largest epsilon ≤ candidate_epsilon that fits the budget,
    and σ is the corresponding noise scale.

    Binary-searches over σ directly instead of ε to avoid calling
    ``calibrate_sigma`` (itself a binary search) inside the outer loop.

    When *clipping_mode* is ``"per_example"``, *sampling_rate* (batch_size /
    dataset_size) is used for the DP-SGD RDP cost formula.

    *num_steps* is the number of optimization steps per round (e.g.
    ``local_epochs × len(trainloader)``).  For ``per_update`` mode this is
    always 1; for ``per_example`` (DP-SGD) it must be passed to account for
    the accumulated per-step RDP cost within a round.

    Returns (-1.0, 0.0) if even *epsilon_min* would exceed the remaining budget,
    signalling that the client's privacy budget is exhausted.
    """
    if clipping_mode == "per_example":
        if sampling_rate is None:
            raise ValueError("sampling_rate is required for per_example mode")
        _sr: float = sampling_rate
    else:
        _sr = 0.0  # unused placeholder; only accessed in per_example branches

    def _calibrate(eps: float) -> float:
        if clipping_mode == "per_example":
            return calibrate_sigma_dp_sgd(eps, _sr, delta)
        return calibrate_sigma(eps, clipping_norm, delta)

    if candidate_epsilon <= 0:
        return -1.0, 0.0

    if candidate_epsilon <= epsilon_min:
        if _is_epsilon_within_budget(
            candidate_epsilon,
            current_rdp,
            epsilon_budget,
            clipping_norm,
            delta,
            clipping_mode=clipping_mode,
            num_steps=num_steps,
            sampling_rate=sampling_rate,
        ):
            sigma = _calibrate(candidate_epsilon)
            return candidate_epsilon, sigma
        return -1.0, 0.0

    sigma_candidate = _calibrate(candidate_epsilon)
    hypothetical = _hypothetical_epsilon(
        current_rdp,
        sigma_candidate,
        clipping_norm,
        delta,
        clipping_mode=clipping_mode,
        num_steps=num_steps,
        sampling_rate=sampling_rate,
    )

    if hypothetical <= epsilon_budget:
        return candidate_epsilon, sigma_candidate

    if not _is_epsilon_within_budget(
        epsilon_min,
        current_rdp,
        epsilon_budget,
        clipping_norm,
        delta,
        clipping_mode=clipping_mode,
        num_steps=num_steps,
        sampling_rate=sampling_rate,
    ):
        return -1.0, 0.0

    # Binary search over σ (monotonic: larger σ → smaller ε → lower RDP cost).
    sigma_min_eps = _calibrate(epsilon_min)
    lo_sigma, hi_sigma = sigma_candidate, sigma_min_eps

    for _ in range(30):
        mid_sigma = (lo_sigma + hi_sigma) / 2.0
        projected = _hypothetical_epsilon(
            current_rdp,
            mid_sigma,
            clipping_norm,
            delta,
            clipping_mode=clipping_mode,
            num_steps=num_steps,
            sampling_rate=sampling_rate,
        )
        if projected <= epsilon_budget:
            hi_sigma = mid_sigma  # this σ fits; try smaller σ (larger ε)
        else:
            lo_sigma = mid_sigma  # too expensive; need larger σ

    result_sigma = (lo_sigma + hi_sigma) / 2.0
    if clipping_mode == "per_example":
        result_epsilon = _rdp_epsilon_for_sigma_dp_sgd(
            result_sigma,
            _sr,
            delta,
        )
    else:
        result_epsilon = _rdp_epsilon_for_sigma(result_sigma, clipping_norm, delta)
    return result_epsilon, result_sigma


def _rdp_cost_at_alpha_for_sigma(
    sigma: float,
    alpha: float,
    clipping_norm: float,
    clipping_mode: str = "per_update",
    sampling_rate: float | None = None,
) -> float:
    """Compute RDP cost at a fixed alpha for a given sigma."""
    if clipping_mode == "per_example":
        if sampling_rate is None:
            raise ValueError("sampling_rate required for per_example mode")
        return compute_rdp_cost_dp_sgd(alpha, sigma, sampling_rate)
    return compute_rdp_cost(alpha, sigma, clipping_norm)


def enforce_rdp_budget(
    candidate_rdp: float,
    current_rdp_sum: float,
    rdp_budget: float,
    rdp_min: float,
    alpha: float,
    clipping_norm: float,
    clipping_mode: str = "per_update",
    num_steps: int = 1,
    sampling_rate: float | None = None,
) -> tuple[float, float]:
    """Return (rdp_cost, sigma) where rdp_cost is the largest RDP cost <= candidate_rdp
    that fits within the remaining RDP budget, and sigma is the corresponding noise scale.

    This operates entirely in RDP -- no conversion to epsilon. Budget comparison is
    linear: current_rdp_sum + rdp_cost * num_steps <= rdp_budget.

    When *clipping_mode* is ``"per_example"``, *clipping_norm* is interpreted as
    the sampling rate and the DP-SGD RDP cost formula is used.

    Returns (-1.0, 0.0) if even *rdp_min* would exceed the remaining budget.
    """
    if candidate_rdp <= 0:
        return -1.0, 0.0

    remaining = rdp_budget - current_rdp_sum
    if remaining <= 0:
        return -1.0, 0.0

    if clipping_mode == "per_example":
        if sampling_rate is None:
            raise ValueError("sampling_rate required for per_example mode")

        def calibrate(r: float) -> float:
            return calibrate_sigma_rdp_dp_sgd(r, alpha, sampling_rate)
    else:

        def calibrate(r: float) -> float:
            return calibrate_sigma_rdp(r, alpha, clipping_norm)

    # Check if candidate fits
    projected = current_rdp_sum + candidate_rdp * num_steps
    if projected <= rdp_budget:
        sigma = calibrate(candidate_rdp)
        return candidate_rdp, sigma

    # Check if even rdp_min fits
    projected_min = current_rdp_sum + rdp_min * num_steps
    if projected_min > rdp_budget:
        return -1.0, 0.0

    # Binary search for the largest rdp_cost that fits
    max_allowed_rdp = remaining / num_steps
    lo, hi = rdp_min, min(candidate_rdp, max_allowed_rdp)

    for _ in range(30):
        mid = (lo + hi) / 2.0
        projected_mid = current_rdp_sum + mid * num_steps
        if projected_mid <= rdp_budget:
            lo = mid  # this fits; try larger
        else:
            hi = mid  # too expensive; need smaller

    result_rdp = (lo + hi) / 2.0
    result_sigma = calibrate(result_rdp)
    return result_rdp, result_sigma
