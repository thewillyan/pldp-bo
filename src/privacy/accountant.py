from __future__ import annotations

import json
import math

import numpy as np

from src.privacy.constants import RDP_ALPHAS
from src.privacy.per_update_dp import compute_rdp_cost, compute_rdp_cost_dp_sgd


class RDPAccountant:
    def __init__(self, delta: float = 1e-5) -> None:
        self._delta = delta
        self._rdp_per_alpha: np.ndarray = np.zeros_like(RDP_ALPHAS)
        self._steps: list[dict] = []

    def step(self, *, sigma: float, clipping_norm: float, num_steps: int = 1,
             mode: str = "per_update") -> None:
        if mode == "per_example":
            cost = np.array(
                [compute_rdp_cost_dp_sgd(float(a), sigma, clipping_norm) for a in RDP_ALPHAS],
                dtype=np.float64,
            )
        else:
            cost = np.array(
                [compute_rdp_cost(float(a), sigma, clipping_norm) for a in RDP_ALPHAS],
                dtype=np.float64,
            )
        self._rdp_per_alpha += cost * num_steps
        self._steps.append({
            "sigma": sigma,
            "clipping_norm": clipping_norm,
            "num_steps": num_steps,
            "mode": mode,
        })

    def get_epsilon(self, delta: float | None = None) -> float:
        if not self._steps:
            return 0.0
        delta_val = delta if delta is not None else self._delta
        log_one_over_delta = math.log(1.0 / delta_val)
        with np.errstate(divide="ignore", invalid="ignore"):
            epsilons = self._rdp_per_alpha + log_one_over_delta / (RDP_ALPHAS - 1.0)
        valid = np.isfinite(epsilons)
        if not valid.any():
            return float("inf")
        return float(np.min(epsilons[valid]))

    def get_epsilon_with_diagnostics(self, delta: float | None = None) -> tuple[float, float]:
        if not self._steps:
            return 0.0, 0.0
        delta_val = delta if delta is not None else self._delta
        log_one_over_delta = math.log(1.0 / delta_val)
        with np.errstate(divide="ignore", invalid="ignore"):
            epsilons = self._rdp_per_alpha + log_one_over_delta / (RDP_ALPHAS - 1.0)
        valid = np.isfinite(epsilons)
        if not valid.any():
            return float("inf"), 0.0
        masked = np.where(valid, epsilons, np.inf)
        best_idx = int(np.argmin(masked))
        return float(epsilons[best_idx]), float(RDP_ALPHAS[best_idx])

    def get_privacy_spent(self, delta: float | None = None) -> dict[str, float]:
        eps = self.get_epsilon(delta)
        return {"epsilon": eps, "delta": delta if delta is not None else self._delta}

    def total_steps(self) -> int:
        return sum(s["num_steps"] for s in self._steps)

    @property
    def rdp_per_alpha(self) -> np.ndarray:
        return self._rdp_per_alpha.copy()

    def reset(self) -> None:
        self._rdp_per_alpha = np.zeros_like(RDP_ALPHAS)
        self._steps = []

    def get_state(self) -> dict:
        return {
            "delta": self._delta,
            "steps": json.dumps(self._steps),
            "rdp_per_alpha": self._rdp_per_alpha.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict) -> RDPAccountant:
        accountant = cls(delta=state["delta"])
        rdp_data = state.get("rdp_per_alpha", [])
        accountant._rdp_per_alpha = np.array(rdp_data, dtype=np.float64)
        if accountant._rdp_per_alpha.shape != RDP_ALPHAS.shape:
            accountant._rdp_per_alpha = np.zeros_like(RDP_ALPHAS)
        steps_raw = state.get("steps", "[]")
        steps_data = json.loads(steps_raw) if isinstance(steps_raw, str) else steps_raw
        for step_info in steps_data:
            accountant._steps.append({
                "sigma": step_info["sigma"],
                "clipping_norm": step_info["clipping_norm"],
                "num_steps": step_info["num_steps"],
                "mode": step_info.get("mode", "per_update"),
            })
        return accountant
