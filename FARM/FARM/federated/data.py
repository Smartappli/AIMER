from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class FederatedDataset:
    """Container for task datasets used by federated clients."""

    train: Any
    validation: Any | None = None
    test: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
