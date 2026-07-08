from __future__ import annotations

import json

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel, Matern, WhiteKernel

from src.privacy.epsilon_scheduler import EpsilonScheduler
from src.utils import deserialize_rng, serialize_rng  # noqa: I001


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    f_best: float,
) -> np.ndarray:
    std = np.maximum(std, 1e-12)
    improvement = f_best - mean
    z = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return np.maximum(ei, 0.0)


def normalize_ei(ei: np.ndarray) -> np.ndarray:
    ei_min, ei_max = ei.min(), ei.max()
    if ei_max > ei_min:
        return (ei - ei_min) / (ei_max - ei_min)
    return np.zeros_like(ei)


def _build_kernel(name: str = "matern52", noise_level: float = 0.01) -> Kernel:
    if name == "matern52":
        return Matern(nu=2.5) + WhiteKernel(noise_level=noise_level)
    if name == "rbf":
        return RBF() + WhiteKernel(noise_level=noise_level)
    if name == "matern32":
        return Matern(nu=1.5) + WhiteKernel(noise_level=noise_level)
    raise ValueError(f"Unknown kernel: {name}")


class PLDPBOScheduler(EpsilonScheduler):
    def __init__(
        self,
        epsilon_min: float,
        epsilon_max: float,
        warmup_rounds: int = 20,
        acquisition_penalty: float = 0.1,
        grid_points: int = 100,
        gp_kernel: str = "matern52",
        observation_noise: float = 0.01,
        seed: int | None = None,
    ) -> None:
        if epsilon_min <= 0:
            raise ValueError("epsilon_min must be positive")
        if epsilon_max <= epsilon_min:
            raise ValueError("epsilon_max must be greater than epsilon_min")
        if warmup_rounds < 2:
            raise ValueError("warmup_rounds must be at least 2")
        if acquisition_penalty < 0:
            raise ValueError("acquisition_penalty must be non-negative")
        if grid_points < 10:
            raise ValueError("grid_points must be at least 10")

        self._epsilon_min = epsilon_min
        self._epsilon_max = epsilon_max
        self._warmup_rounds = warmup_rounds
        self._acquisition_penalty = acquisition_penalty
        self._grid_points = grid_points
        self._gp_kernel_name = gp_kernel
        self._observation_noise = observation_noise
        self._rng = np.random.RandomState(seed)

        self._warmup_epsilons = np.linspace(
            epsilon_min, epsilon_max, warmup_rounds,
        )
        self._phase: str = "warmup"
        self._round: int = 0
        self._observations: list[tuple[float, float]] = []
        self._gp: GaussianProcessRegressor | None = None
        self._f_best: float = float("inf")
        self._remaining_budget: float | None = None

    def set_remaining_budget(self, remaining: float | None) -> None:
        self._remaining_budget = remaining

    def get_epsilon(self) -> float:
        if self._phase == "warmup":
            if self._round >= self._warmup_rounds:
                return self._select_bo_epsilon()
            return float(self._warmup_epsilons[self._round])
        return self._select_bo_epsilon()

    def step(self, epsilon: float, metric: float) -> None:
        self._observations.append((epsilon, metric))
        self._f_best = min(self._f_best, metric)

        if self._phase == "warmup" and len(self._observations) >= self._warmup_rounds:
            self._fit_gp()
            self._phase = "bo"
        elif self._phase == "bo":
            self._fit_gp()

        self._round += 1

    def _fit_gp(self) -> None:
        x = np.array([[eps] for eps, _ in self._observations])
        y = np.array([m for _, m in self._observations])
        kernel = _build_kernel(self._gp_kernel_name, self._observation_noise)
        if self._gp is not None:
            # carry forward kernel hyperparameters (length-scale, noise-level)
            # learned from previous fits, providing warm-start continuity
            # across sequential BO rounds
            kernel = self._gp.kernel_
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            random_state=self._rng.randint(0, 2**31),
            normalize_y=True,
        )
        self._gp.fit(x, y)

    def _select_bo_epsilon(self) -> float:
        if self._gp is None:
            return float(self._rng.uniform(self._epsilon_min, self._epsilon_max))

        grid = np.linspace(self._epsilon_min, self._epsilon_max, self._grid_points)
        mean, std = self._gp.predict(grid.reshape(-1, 1), return_std=True)

        ei = expected_improvement(mean, std, self._f_best)
        ei_norm = normalize_ei(ei)

        penalty = (grid - self._epsilon_min) / (self._epsilon_max - self._epsilon_min)
        alpha = ei_norm - self._acquisition_penalty * penalty

        # Mask grid points that would exceed the remaining privacy budget
        if self._remaining_budget is not None:
            alpha[grid > self._remaining_budget] = -np.inf

        # Degenerate case (all ei_norm equal → all zeros): alpha = -λ · penalty,
        # which automatically selects epsilon_min as argmax.
        return float(grid[np.argmax(alpha)])

    def get_state(self) -> dict:
        return {
            "type": "pldp_bo",
            "epsilon_min": self._epsilon_min,
            "epsilon_max": self._epsilon_max,
            "warmup_rounds": self._warmup_rounds,
            "acquisition_penalty": self._acquisition_penalty,
            "grid_points": self._grid_points,
            "gp_kernel": self._gp_kernel_name,
            "observation_noise": self._observation_noise,
            "phase": self._phase,
            "round": self._round,
            "observations": json.dumps(self._observations),
            "f_best": self._f_best,
            "rng_state": serialize_rng(self._rng),
            "remaining_budget": self._remaining_budget,
        }

    @classmethod
    def from_state(cls, state: dict) -> PLDPBOScheduler:
        scheduler = cls(
            epsilon_min=state["epsilon_min"],
            epsilon_max=state["epsilon_max"],
            warmup_rounds=state["warmup_rounds"],
            acquisition_penalty=state["acquisition_penalty"],
            grid_points=state["grid_points"],
            gp_kernel=state["gp_kernel"],
            observation_noise=state["observation_noise"],
        )
        scheduler._phase = state["phase"]
        scheduler._round = state["round"]
        scheduler._observations = [tuple(obs) for obs in json.loads(state["observations"])]
        scheduler._f_best = state["f_best"]
        if "rng_state" in state:
            scheduler._rng.set_state(deserialize_rng(state["rng_state"]))
        if "remaining_budget" in state:
            scheduler._remaining_budget = state["remaining_budget"]
        if scheduler._phase == "bo":
            scheduler._fit_gp()
        return scheduler

    def __repr__(self) -> str:
        return (
            f"PLDPBOScheduler(phase={self._phase}, round={self._round}, "
            f"obs={len(self._observations)})"
        )
