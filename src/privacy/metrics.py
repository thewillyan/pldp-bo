from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def compute_utility_loss(
    model: nn.Module,
    valloader: DataLoader,
    criterion: nn.Module | None = None,
) -> float:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for images, labels in valloader:
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            num_batches += 1
    return total_loss / max(num_batches, 1)
