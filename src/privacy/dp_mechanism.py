from __future__ import annotations

import numpy as np
import torch
from torch import nn


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    return total_norm


def calibrate_gaussian_noise(
    target_epsilon: float,
    delta: float,
    sensitivity: float = 1.0,
    sampling_rate: float = 1.0,
    steps: int = 1,
) -> float:
    if target_epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")

    q = sampling_rate
    eps_target = target_epsilon

    def _compute_rdp_eps(sigma: float) -> float:
        alpha = np.arange(2, 64, 0.5)
        rdp = alpha / (2 * sigma ** 2)
        eps = rdp + np.log(1 / delta) / (alpha - 1)
        return np.min(eps) * q * steps

    lo, hi = 0.1, 100.0
    for _ in range(50):
        mid = (lo + hi) / 2
        eps = _compute_rdp_eps(mid)
        if eps > eps_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
