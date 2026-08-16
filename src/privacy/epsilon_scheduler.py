from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.utils import deserialize_rng, serialize_rng


class EpsilonScheduler(ABC):
    @abstractmethod
    def get_epsilon(self) -> float: ...

    def step(self, epsilon: float, metric: float) -> None:  # noqa: B027
        pass

    @abstractmethod
    def set_remaining_budget(self, remaining: float | None) -> None:
        pass

    @abstractmethod
    def get_state(self) -> dict: ...

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict) -> EpsilonScheduler: ...


class FixedEpsilonScheduler(EpsilonScheduler):
    def __init__(self, epsilon: float) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self._epsilon = epsilon

    def set_remaining_budget(self, remaining: float | None) -> None:
        pass

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
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    def set_remaining_budget(self, remaining: float | None) -> None:
        pass

    def get_epsilon(self) -> float:
        return float(
            self._rng.uniform(self._epsilon_min, self._epsilon_max),
        )

    def get_state(self) -> dict:
        state = {
            "type": "uniform_random",
            "epsilon_min": self._epsilon_min,
            "epsilon_max": self._epsilon_max,
            "rng_state": serialize_rng(self._rng),
        }
        if self._seed is not None:
            state["seed"] = self._seed
        return state

    @classmethod
    def from_state(cls, state: dict) -> UniformRandomEpsilonScheduler:
        scheduler = cls(
            epsilon_min=state["epsilon_min"],
            epsilon_max=state["epsilon_max"],
            seed=state.get("seed"),
        )
        if "rng_state" in state:
            scheduler._rng.set_state(deserialize_rng(state["rng_state"]))
        return scheduler

    def __repr__(self) -> str:
        return f"UniformRandomEpsilonScheduler(min={self._epsilon_min}, max={self._epsilon_max})"


class RDPNativeScheduler(ABC):
    """Abstract base for RDP-native schedulers (analogous to EpsilonScheduler)."""

    @abstractmethod
    def get_rdp(self) -> float: ...

    def step(self, rdp: float, metric: float) -> None:  # noqa: B027
        pass

    @abstractmethod
    def set_remaining_budget(self, remaining: float | None) -> None:
        pass

    @abstractmethod
    def get_state(self) -> dict: ...

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict) -> RDPNativeScheduler: ...


class FixedRDPScheduler(RDPNativeScheduler):
    def __init__(self, rdp_target: float) -> None:
        if rdp_target <= 0:
            raise ValueError("rdp_target must be positive")
        self._rdp_target = rdp_target

    def set_remaining_budget(self, remaining: float | None) -> None:
        pass

    def get_rdp(self) -> float:
        return self._rdp_target

    def get_state(self) -> dict:
        return {"type": "fixed_rdp", "rdp_target": self._rdp_target}

    @classmethod
    def from_state(cls, state: dict) -> FixedRDPScheduler:
        return cls(rdp_target=state["rdp_target"])

    def __repr__(self) -> str:
        return f"FixedRDPScheduler(rdp_target={self._rdp_target})"


class UniformRandomRDPScheduler(RDPNativeScheduler):
    def __init__(self, rdp_min: float, rdp_max: float, seed: int | None = None) -> None:
        if rdp_min <= 0:
            raise ValueError("rdp_min must be positive")
        if rdp_max <= rdp_min:
            raise ValueError("rdp_max must be greater than rdp_min")
        self._rdp_min = rdp_min
        self._rdp_max = rdp_max
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    def set_remaining_budget(self, remaining: float | None) -> None:
        pass

    def get_rdp(self) -> float:
        return float(self._rng.uniform(self._rdp_min, self._rdp_max))

    def get_state(self) -> dict:
        state = {
            "type": "uniform_random_rdp",
            "rdp_min": self._rdp_min,
            "rdp_max": self._rdp_max,
            "rng_state": serialize_rng(self._rng),
        }
        if self._seed is not None:
            state["seed"] = self._seed
        return state

    @classmethod
    def from_state(cls, state: dict) -> UniformRandomRDPScheduler:
        scheduler = cls(
            rdp_min=state["rdp_min"],
            rdp_max=state["rdp_max"],
            seed=state.get("seed"),
        )
        if "rng_state" in state:
            scheduler._rng.set_state(deserialize_rng(state["rng_state"]))
        return scheduler

    def __repr__(self) -> str:
        return f"UniformRandomRDPScheduler(min={self._rdp_min}, max={self._rdp_max})"
