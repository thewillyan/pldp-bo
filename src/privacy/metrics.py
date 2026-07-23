from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.device import to_device

logger = logging.getLogger(__name__)


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
    num_batches = 0
    all_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_images, batch_labels in valloader:
            images, labels = to_device((batch_images, batch_labels))
            outputs = model(images)
            clipped = torch.clamp(outputs, min=-20.0, max=20.0)
            if outputs.is_floating_point() and (outputs != clipped).any():
                logger.warning(
                    "compute_validation_stats: clipped %d/%d logits to [-20, 20]",
                    (outputs != clipped).sum().item(), outputs.numel(),
                )
            outputs = clipped
            total_loss += criterion(outputs, labels).item()
            all_logits.append(outputs.cpu())
            num_batches += 1
    avg_loss = total_loss / max(num_batches, 1)
    logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0)
    return avg_loss, logits
