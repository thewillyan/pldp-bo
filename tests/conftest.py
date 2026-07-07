from __future__ import annotations

import pytest

from src.utils import set_seed


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    set_seed(42)
