from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseModel

_INPUT_SIZE_MAP: dict[str, int] = {
    "mnist": 28 * 28,
    "cifar10": 3 * 32 * 32,
    "cifar100": 3 * 32 * 32,
    "femnist": 28 * 28,
}


class MLP(nn.Module):
    def __init__(self, num_classes: int = 10, input_size: int = 28 * 28) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = x.view(x.size(0), -1)
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        out: torch.Tensor = self.fc3(h)
        return out


class MLPModel(BaseModel):
    def __init__(self, num_classes: int = 10, dataset_name: str | None = None) -> None:
        super().__init__()
        input_size = _INPUT_SIZE_MAP.get(dataset_name or "mnist", 28 * 28)
        self._model = MLP(num_classes, input_size=input_size)

    def get_model(self) -> nn.Module:
        return self._model
