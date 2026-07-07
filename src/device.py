from __future__ import annotations

import torch

_DEVICE: torch.device | None = None


def get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def to_device(batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    device = get_device()
    return (batch[0].to(device), batch[1].to(device))
