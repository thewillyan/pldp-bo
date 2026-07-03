from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoundMetrics:
    round_num: int
    server_loss: float | None = None
    accuracy: float | None = None
    epsilon: float | None = None
    num_clients: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClientMetrics:
    client_id: int
    loss: float | None = None
    accuracy: float | None = None
    epsilon: float | None = None
    num_samples: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
