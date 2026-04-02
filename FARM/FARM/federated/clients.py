"""Federated client implementations for task and RAG workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import flwr as fl

from .rag import parameters_to_state, state_to_parameters

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .rag import RagIndex, RagState
    from .tasks import EvaluationResult, TaskDefinition, TrainingResult


@dataclass(slots=True)
class ClientContext:
    """Context shared with training hooks."""

    metadata: Mapping[str, Any]


class FederatedTaskClient(fl.client.NumPyClient):
    """Flower NumPyClient backed by a TaskDefinition."""

    def __init__(
        self,
        task: TaskDefinition,
        context: ClientContext | None = None,
    ) -> None:
        """Initialize a task-oriented federated client."""
        self._task = task
        self._context = context or ClientContext(metadata={})

    def get_parameters(
        self,
        _config: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        """Return current model parameters from task handlers.

        Returns:
            Current model parameters formatted for Flower transport.

        """
        return self._task.handlers.get_parameters(self._task.model)

    def fit(
        self,
        parameters: Sequence[Any],
        config: Mapping[str, Any],
    ) -> tuple[list[Any], int, dict[str, Any]]:
        """Apply incoming parameters and execute one local training step.

        Returns:
            Tuple ``(parameters, num_examples, metrics)`` after local training.

        """
        self._task.handlers.set_parameters(self._task.model, parameters)
        result: TrainingResult = self._task.handlers.train(
            self._task.model,
            self._task.dataset,
            config,
        )
        return result.parameters, result.num_examples, dict(result.metrics)

    def evaluate(
        self,
        parameters: Sequence[Any],
        config: Mapping[str, Any],
    ) -> tuple[float, int, dict[str, Any]]:
        """Evaluate the model using task-provided evaluation logic.

        Returns:
            Tuple ``(loss, num_examples, metrics)`` from local evaluation.

        """
        self._task.handlers.set_parameters(self._task.model, parameters)
        result: EvaluationResult = self._task.handlers.evaluate(
            self._task.model,
            self._task.dataset,
            config,
        )
        return result.loss, result.num_examples, dict(result.metrics)


class RagClient(fl.client.Client):
    """Client that shares RAG index updates over Flower parameters."""

    def __init__(
        self,
        index: RagIndex,
        document_provider: Callable[[Mapping[str, Any]], Sequence[RagState]],
    ) -> None:
        """Initialize a RAG synchronization client."""
        self._index = index
        self._document_provider = document_provider

    def get_parameters(
        self,
        _ins: fl.common.GetParametersIns,
    ) -> fl.common.GetParametersRes:
        """Return the current serialized RAG index state.

        Returns:
            Flower response containing serialized RAG state parameters.

        """
        parameters = state_to_parameters(self._index.to_state())
        return fl.common.GetParametersRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="ok"),
            parameters=parameters,
        )

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        """Merge incoming state, apply local updates, and return new state.

        Returns:
            Flower fit response with updated serialized state and metrics.

        """
        incoming = parameters_to_state(ins.parameters)
        if incoming:
            self._index.merge_state(incoming)

        updates = self._document_provider(ins.config)
        for update in updates:
            self._index.merge_state(update)

        parameters = state_to_parameters(self._index.to_state())
        return fl.common.FitRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="ok"),
            parameters=parameters,
            num_examples=self._index.size,
            metrics={"rag_docs": self._index.size},
        )

    def evaluate(self, _ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        """Return a lightweight evaluation payload for compatibility.

        Returns:
            Flower evaluation response with zero loss and RAG doc metrics.

        """
        return fl.common.EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="ok"),
            loss=0.0,
            num_examples=0,
            metrics={"rag_docs": self._index.size},
        )
