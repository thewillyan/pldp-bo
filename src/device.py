from __future__ import annotations

import functools

import torch


@functools.cache
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    device = get_device()
    return (batch[0].to(device), batch[1].to(device))
