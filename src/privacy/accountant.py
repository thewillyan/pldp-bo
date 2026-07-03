from __future__ import annotations

from typing import Optional

import numpy as np
from opacus.accountants import RDPAccountant as OpacusRDPAccountant


class RDPAccountant:
    def __init__(self, delta: float = 1e-5):
        self._accountant = OpacusRDPAccountant()
        self._delta = delta
        self._steps: list[dict] = []

    def step(self, *, noise_multiplier: float, sample_rate: float, num_steps: int = 1) -> None:
        for _ in range(num_steps):
            self._accountant.step(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
            )
        self._steps.append({
            "noise_multiplier": noise_multiplier,
            "sample_rate": sample_rate,
            "num_steps": num_steps,
        })

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        return self._accountant.get_epsilon(delta=delta or self._delta)

    def get_privacy_spent(self, delta: Optional[float] = None) -> dict[str, float]:
        eps = self.get_epsilon(delta)
        return {"epsilon": eps, "delta": delta or self._delta}

    def total_steps(self) -> int:
        return sum(s["num_steps"] for s in self._steps)

    def reset(self) -> None:
        self._accountant = OpacusRDPAccountant()
        self._steps = []

    def get_state(self) -> dict:
        return {"delta": self._delta, "steps": list(self._steps)}

    @classmethod
    def from_state(cls, state: dict) -> RDPAccountant:
        accountant = cls(delta=state["delta"])
        for step_info in state.get("steps", []):
            accountant.step(
                noise_multiplier=step_info["noise_multiplier"],
                sample_rate=step_info["sample_rate"],
                num_steps=step_info["num_steps"],
            )
        return accountant
