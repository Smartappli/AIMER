from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import flwr as fl

from .rag import RagIndex, parameters_to_state, state_to_parameters


class TaskFedAvgStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy with optional task metadata."""

    def __init__(
        self,
        task_name: str,
        on_fit_config_fn: Callable[[int], Mapping[str, Any]] | None = None,
        on_evaluate_config_fn: Callable[[int], Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            **kwargs,
        )
        self.task_name = task_name


class RagFedAvgStrategy(fl.server.strategy.FedAvg):
    """FedAvg-like strategy that merges RAG documents instead of averaging weights."""

    def __init__(
        self,
        rag_index: RagIndex,
        on_fit_config_fn: Callable[[int], Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(on_fit_config_fn=on_fit_config_fn, **kwargs)
        self._rag_index = rag_index

    def initialize_parameters(
        self,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> fl.common.Parameters | None:
        return state_to_parameters(self._rag_index.to_state())

    def aggregate_fit(
        self,
        server_round: int,
        results: Sequence[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: Sequence[BaseException] | None,
    ) -> tuple[fl.common.Parameters | None, Mapping[str, Any]]:
        for _, fit_res in results:
            state = parameters_to_state(fit_res.parameters)
            if state:
                self._rag_index.merge_state(state)

        parameters = state_to_parameters(self._rag_index.to_state())
        metrics = {"rag_docs": self._rag_index.size, "round": server_round}
        return parameters, metrics
