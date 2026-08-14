from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.device import to_device


def compute_utility_loss(
    model: nn.Module,
    valloader: DataLoader,
    criterion: nn.Module | None = None,
) -> float:
    """Compute average validation loss.

    Parameters
    ----------
        criterion: Loss function that returns a scalar tensor when called
            with (output, target). Defaults to CrossEntropyLoss.
    """
    loss, _ = compute_validation_stats(model, valloader, criterion)
    return loss


def compute_validation_stats(
    model: nn.Module,
    valloader: DataLoader,
    criterion: nn.Module | None = None,
) -> tuple[float, torch.Tensor]:
    """Compute average validation loss and return all logits.

    Returns
    -------
        (avg_loss, logits) where logits is a 2-D tensor of shape
        (num_samples, num_classes) concatenated from all batches.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_images, batch_labels in valloader:
            images, labels = to_device((batch_images, batch_labels))
            outputs = model(images)
            batch_size = images.size(0)
            total_loss += criterion(outputs, labels).item() * batch_size
            total_samples += batch_size
            all_logits.append(outputs.cpu())
    avg_loss = total_loss / max(total_samples, 1)
    logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, 0)
    return avg_loss, logits
