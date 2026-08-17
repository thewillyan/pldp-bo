from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset

_PROCESSED_DIR = "FEMNIST/processed"
TRAIN_FILE = "femnist_train.pt"
TEST_FILE = "femnist_test.pt"
USER_KEYS_FILE = "femnist_user_keys.pt"

_MISSING_MSG = (
    "FEMNIST data not found at {path}. Get the files from "
    "data/femnist-dataset-PyTorch/femnist.tar.gz, extract them, and place "
    "femnist_train.pt, femnist_test.pt and femnist_user_keys.pt under "
    "FEMNIST/processed/ (the pipeline never downloads data)."
)


class FEMNISTDataset(Dataset[tuple[Tensor, int]]):
    """LEAF FEMNIST with per-sample writer ids (spec §9.9).

    ``femnist_train.pt`` / ``femnist_test.pt`` each hold ``[data, targets,
    users]`` where ``users[i]`` indexes ``femnist_user_keys.pt`` (the writer
    of sample ``i``). Images are 28x28 grayscale; 62 classes. Data is loaded
    as float and rescaled to [0, 1] when stored in 0-255 format (the stored
    scale differs across provenance, so the loader normalizes robustly).
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable[[Tensor], Tensor] | None = None,
        download: bool = False,
    ) -> None:
        processed = Path(root) / _PROCESSED_DIR
        data_file = processed / (TRAIN_FILE if train else TEST_FILE)
        keys_file = processed / USER_KEYS_FILE
        if not data_file.is_file() or not keys_file.is_file():
            raise FileNotFoundError(
                _MISSING_MSG.format(path=processed),
            )
        data_targets_users = torch.load(data_file, weights_only=False)
        if not isinstance(data_targets_users, (list, tuple)) or len(data_targets_users) < 3:
            raise ValueError(
                f"FEMNIST file {data_file} must hold [data, targets, users], "
                f"got {type(data_targets_users).__name__}",
            )
        data, targets, users = data_targets_users[0], data_targets_users[1], data_targets_users[2]
        self.data = torch.as_tensor(data).float()
        if self.data.max() > 1.0:
            self.data = self.data / 255.0
        self.targets = torch.as_tensor(targets)
        self.users = torch.as_tensor(users)
        self.user_keys = torch.load(keys_file, weights_only=False)
        self._transform: Callable[[Tensor], Tensor] | None = transform
        self._download = download  # accepted for torchvision-compatible signature; never downloads

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        img = self.data[idx].unsqueeze(0)
        if self._transform is not None:
            img = self._transform(img)
        return img, int(self.targets[idx])


def femnist_counts(root: str) -> tuple[int, int, int]:
    """Return (n_train, n_test, n_writers) re-read from the .pt files.

    Verification-only helper for the §5.1 extraction check (spec §9.9);
    expected values ≈654,281 / ≈163,570 / 3,597.
    """
    processed = Path(root) / _PROCESSED_DIR
    train = torch.load(processed / TRAIN_FILE, weights_only=False)
    test = torch.load(processed / TEST_FILE, weights_only=False)
    keys = torch.load(processed / USER_KEYS_FILE, weights_only=False)
    if isinstance(keys, dict) and "users" in keys:
        keys = keys["users"]
    return len(train[0]), len(test[0]), len(keys)
