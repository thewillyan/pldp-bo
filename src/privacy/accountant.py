from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.privacy.per_update_dp import compute_rdp_cost

_RDP_ALPHAS: np.ndarray = np.arange(2, 65, dtype=np.float64)


class RDPAccountant:
    def __init__(self, delta: float = 1e-5):
        self._delta = delta
        self._rdp_per_alpha: np.ndarray = np.zeros_like(_RDP_ALPHAS)
        self._steps: list[dict] = []

    def step(self, *, sigma: float, clipping_norm: float, num_steps: int = 1) -> None:
        cost = np.array(
            [compute_rdp_cost(float(a), sigma, clipping_norm) for a in _RDP_ALPHAS],
            dtype=np.float64,
        )
        self._rdp_per_alpha += cost * num_steps
        self._steps.append({
            "sigma": sigma,
            "clipping_norm": clipping_norm,
            "num_steps": num_steps,
        })

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        if not self._steps:
            return 0.0
        delta_val = delta if delta is not None else self._delta
        log_one_over_delta = math.log(1.0 / delta_val)
        with np.errstate(divide="ignore", invalid="ignore"):
            epsilons = self._rdp_per_alpha + log_one_over_delta / (_RDP_ALPHAS - 1.0)
        valid = np.isfinite(epsilons)
        if not valid.any():
            return 0.0
        return float(np.min(epsilons[valid]))

    def get_privacy_spent(self, delta: Optional[float] = None) -> dict[str, float]:
        eps = self.get_epsilon(delta)
        return {"epsilon": eps, "delta": delta if delta is not None else self._delta}

    def total_steps(self) -> int:
        return sum(s["num_steps"] for s in self._steps)

    def reset(self) -> None:
        self._rdp_per_alpha = np.zeros_like(_RDP_ALPHAS)
        self._steps = []

    def get_state(self) -> dict:
        return {
            "delta": self._delta,
            "steps": list(self._steps),
            "rdp_per_alpha": self._rdp_per_alpha.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict) -> RDPAccountant:
        accountant = cls(delta=state["delta"])
        rdp_data = state.get("rdp_per_alpha", [])
        accountant._rdp_per_alpha = np.array(rdp_data, dtype=np.float64)
        if accountant._rdp_per_alpha.shape != _RDP_ALPHAS.shape:
            accountant._rdp_per_alpha = np.zeros_like(_RDP_ALPHAS)
        for step_info in state.get("steps", []):
            accountant._steps.append({
                "sigma": step_info["sigma"],
                "clipping_norm": step_info["clipping_norm"],
                "num_steps": step_info["num_steps"],
            })
        return accountant
