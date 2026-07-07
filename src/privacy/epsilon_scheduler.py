from __future__ import annotations

import json
from abc import ABC, abstractmethod

import numpy as np


class EpsilonScheduler(ABC):
    @abstractmethod
    def get_epsilon(self) -> float:
        ...

    def step(self, epsilon: float, metric: float) -> None:  # noqa: B027
        pass

    @abstractmethod
    def get_state(self) -> dict:
        ...

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict) -> EpsilonScheduler:
        ...


class FixedEpsilonScheduler(EpsilonScheduler):
    def __init__(self, epsilon: float) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self._epsilon = epsilon

    def get_epsilon(self) -> float:
        return self._epsilon

    def get_state(self) -> dict:
        return {"type": "fixed", "epsilon": self._epsilon}

    @classmethod
    def from_state(cls, state: dict) -> FixedEpsilonScheduler:
        return cls(epsilon=state["epsilon"])

    def __repr__(self) -> str:
        return f"FixedEpsilonScheduler(epsilon={self._epsilon})"


class UniformRandomEpsilonScheduler(EpsilonScheduler):
    def __init__(self, epsilon_min: float, epsilon_max: float, seed: int | None = None) -> None:
        if epsilon_min <= 0:
            raise ValueError("epsilon_min must be positive")
        if epsilon_max <= epsilon_min:
            raise ValueError("epsilon_max must be greater than epsilon_min")
        self._epsilon_min = epsilon_min
        self._epsilon_max = epsilon_max
        self._rng = np.random.RandomState(seed)

    def get_epsilon(self) -> float:
        return float(
            self._rng.uniform(self._epsilon_min, self._epsilon_max),
        )

    def get_state(self) -> dict:
        rng_state = self._rng.get_state()
        return {
            "type": "uniform_random",
            "epsilon_min": self._epsilon_min,
            "epsilon_max": self._epsilon_max,
            "rng_state": json.dumps([
                x.tolist() if isinstance(x, np.ndarray) else x for x in rng_state
            ]),
        }

    @classmethod
    def from_state(cls, state: dict) -> UniformRandomEpsilonScheduler:
        scheduler = cls(
            epsilon_min=state["epsilon_min"],
            epsilon_max=state["epsilon_max"],
        )
        rng = json.loads(state["rng_state"])
        scheduler._rng.set_state(tuple(
            np.array(x, dtype=np.uint32) if isinstance(x, list) else x for x in rng
        ))
        return scheduler

    def __repr__(self) -> str:
        return f"UniformRandomEpsilonScheduler(min={self._epsilon_min}, max={self._epsilon_max})"
