from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import flwr as fl

from .rag import RagIndex, RagState, parameters_to_state, state_to_parameters
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
        self._task = task
        self._context = context or ClientContext(metadata={})

    def get_parameters(
        self,
        config: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        return self._task.handlers.get_parameters(self._task.model)

    def fit(
        self,
        parameters: Sequence[Any],
        config: Mapping[str, Any],
    ) -> tuple[list[Any], int, dict[str, Any]]:
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
        self._index = index
        self._document_provider = document_provider

    def get_parameters(
        self,
        ins: fl.common.GetParametersIns,
    ) -> fl.common.GetParametersRes:
        parameters = state_to_parameters(self._index.to_state())
        return fl.common.GetParametersRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="ok"),
            parameters=parameters,
        )

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
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

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        return fl.common.EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="ok"),
            loss=0.0,
            num_examples=0,
            metrics={"rag_docs": self._index.size},
        )
