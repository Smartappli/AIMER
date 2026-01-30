from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np

from .data import FederatedDataset


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    RAG = "rag"
    OTHER = "other"


@dataclass(slots=True)
class TrainingResult:
    parameters: list[np.ndarray]
    num_examples: int
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResult:
    loss: float
    num_examples: int
    metrics: Mapping[str, Any] = field(default_factory=dict)


class TaskHandlers(Protocol):
    def get_parameters(self, model: Any) -> list[np.ndarray]:
        ...

    def set_parameters(
        self, model: Any, parameters: Sequence[np.ndarray],
    ) -> None:
        ...

    def train(
        self,
        model: Any,
        dataset: FederatedDataset,
        config: Mapping[str, Any],
    ) -> TrainingResult:
        ...

    def evaluate(
        self,
        model: Any,
        dataset: FederatedDataset,
        config: Mapping[str, Any],
    ) -> EvaluationResult:
        ...


@dataclass(slots=True)
class TaskDefinition:
    name: str
    task_type: TaskType
    model: Any
    dataset: FederatedDataset
    handlers: TaskHandlers
    metadata: Mapping[str, Any] = field(default_factory=dict)
