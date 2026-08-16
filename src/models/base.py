from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

from src.device import get_device


class BaseModel(ABC):
    @abstractmethod
    def get_model(self) -> nn.Module: ...

    def get_weights(self) -> list[np.ndarray]:
        return [val.cpu().numpy().copy() for val in self.get_model().state_dict().values()]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        state_dict = self.get_model().state_dict()
        if len(weights) != len(state_dict):
            raise ValueError(
                f"Expected {len(state_dict)} weight arrays, got {len(weights)}. "
                f"State dict keys: {list(state_dict.keys())}",
            )
        device = get_device()
        new_state = {
            k: torch.tensor(w, dtype=state_dict[k].dtype, device=device)
            for k, w in zip(state_dict.keys(), weights, strict=True)
        }
        self.get_model().load_state_dict(new_state)
