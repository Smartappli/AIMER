from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FederatedDataset:
    """Container for task datasets used by federated clients."""

    train: Any
    validation: Any | None = None
    test: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
