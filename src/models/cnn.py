from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseModel


from src.device import get_device

_INPUT_CHANNELS_MAP: dict[str, int] = {
    "mnist": 1,
    "cifar10": 3,
    "cifar100": 3,
}


class CNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, input_h: int = 32, input_w: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self._fc1_input = self._get_conv_output(in_channels, input_h, input_w)
        self.fc1 = nn.Linear(self._fc1_input, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _get_conv_output(self, in_channels: int, h: int, w: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w, device=get_device())
            x = self.pool(self.relu(self.conv1(dummy)))
            x = self.pool(self.relu(self.conv2(x)))
            return x.numel() // x.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class CNNModel(BaseModel):
    def __init__(self, num_classes: int = 10, dataset_name: str | None = None) -> None:
        super().__init__()
        in_channels = _INPUT_CHANNELS_MAP.get(dataset_name or "cifar10", 3)
        self._model = CNN(num_classes, in_channels=in_channels)

    def get_model(self) -> nn.Module:
        return self._model
