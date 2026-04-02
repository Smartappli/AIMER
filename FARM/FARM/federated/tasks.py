"""Task definitions and protocol contracts for federated workloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np

from .data import FederatedDataset


class TaskType(str, Enum):
    """Supported task categories for federated training."""

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    RAG = "rag"
    OTHER = "other"


@dataclass(slots=True)
class TrainingResult:
    """Result payload returned by a training step."""

    parameters: list[np.ndarray]
    num_examples: int
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResult:
    """Result payload returned by an evaluation step."""

    loss: float
    num_examples: int
    metrics: Mapping[str, Any] = field(default_factory=dict)


class TaskHandlers(Protocol):
    """Protocol that task-specific adapters must implement."""

    def get_parameters(self, model: Any) -> list[np.ndarray]:
        """Extract model parameters as NumPy arrays."""

    def set_parameters(
        self,
        model: Any,
        parameters: Sequence[np.ndarray],
    ) -> None:
        """Load model parameters from a sequence of NumPy arrays."""

    def train(
        self,
        model: Any,
        dataset: FederatedDataset,
        config: Mapping[str, Any],
    ) -> TrainingResult:
        """Run one local training step and return training metadata."""

    def evaluate(
        self,
        model: Any,
        dataset: FederatedDataset,
        config: Mapping[str, Any],
    ) -> EvaluationResult:
        """Run local evaluation and return loss/metrics."""


@dataclass(slots=True)
class TaskDefinition:
    """Description of a federated task and its runtime handlers."""

    name: str
    task_type: TaskType
    model: Any
    dataset: FederatedDataset
    handlers: TaskHandlers
    metadata: Mapping[str, Any] = field(default_factory=dict)
