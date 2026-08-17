from __future__ import annotations

import torch
from torch import nn

from src.device import get_device
from src.models.base import BaseModel

_INPUT_CHANNELS_MAP: dict[str, int] = {
    "mnist": 1,
    "cifar10": 3,
    "cifar100": 3,
    "femnist": 1,
}

_INPUT_DIMS_MAP: dict[str, tuple[int, int]] = {
    "mnist": (28, 28),
    "cifar10": (32, 32),
    "cifar100": (32, 32),
    "femnist": (28, 28),
}


class CNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        input_h: int = 32,
        input_w: int = 32,
    ) -> None:
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
            x: torch.Tensor = self.pool(self.relu(self.conv1(dummy)))
            x = self.pool(self.relu(self.conv2(x)))
            return x.numel() // x.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self.pool(self.relu(self.conv1(x)))
        h = self.pool(self.relu(self.conv2(h)))
        h = h.view(h.size(0), -1)
        h = self.relu(self.fc1(h))
        h = self.dropout(h)
        out: torch.Tensor = self.fc2(h)
        return out


class CNNModel(BaseModel):
    def __init__(self, num_classes: int = 10, dataset_name: str | None = None) -> None:
        super().__init__()
        dataset_name = dataset_name or "cifar10"
        in_channels = _INPUT_CHANNELS_MAP.get(dataset_name, 3)
        input_h, input_w = _INPUT_DIMS_MAP.get(dataset_name, (32, 32))
        self._model = CNN(
            num_classes,
            in_channels=in_channels,
            input_h=input_h,
            input_w=input_w,
        )

    def get_model(self) -> nn.Module:
        return self._model
