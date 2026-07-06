from __future__ import annotations

import math

import numpy as np


def calibrate_sigma(epsilon: float, clipping_norm: float, delta: float) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    if clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive")
    return clipping_norm * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def clip_update(delta: np.ndarray, clipping_norm: float) -> np.ndarray:
    norm = np.linalg.norm(delta)
    if norm > clipping_norm:
        return delta * (clipping_norm / norm)
    return delta


def add_gaussian_noise(clipped_delta: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, size=clipped_delta.shape).astype(clipped_delta.dtype)
    return clipped_delta + noise


def compute_rdp_cost(alpha: float, sigma: float, clipping_norm: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return alpha * clipping_norm**2 / (2.0 * sigma**2)


class PerUpdateGaussianMechanism:
    def __init__(self, clipping_norm: float, delta: float):
        if clipping_norm <= 0:
            raise ValueError("clipping_norm must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        self._clipping_norm = clipping_norm
        self._delta = delta

    def apply(self, delta: np.ndarray, epsilon: float) -> tuple[np.ndarray, float]:
        sigma = calibrate_sigma(epsilon, self._clipping_norm, self._delta)
        clipped = clip_update(delta, self._clipping_norm)
        noisy = add_gaussian_noise(clipped, sigma)
        return noisy, sigma

    @property
    def clipping_norm(self) -> float:
        return self._clipping_norm

    @property
    def delta(self) -> float:
        return self._delta
