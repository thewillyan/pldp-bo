from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseModel


class MLP(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class MLPModel(BaseModel):
    def __init__(self, num_classes: int = 10) -> None:
        self._model = MLP(num_classes)

    def get_model(self) -> nn.Module:
        return self._model
