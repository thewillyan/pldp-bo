from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Set seed for reproducibility across all random sources.

    Args:
        seed: The seed value.
        deterministic: If True, also set cuDNN to deterministic mode
                       (slower but fully reproducible on GPU).
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
