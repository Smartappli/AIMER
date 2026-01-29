from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import flwr as fl

from .clients import FederatedTaskClient, RagClient
from .rag import RagIndex, RagState
from .strategies import RagFedAvgStrategy, TaskFedAvgStrategy
from .tasks import TaskDefinition


@dataclass(slots=True)
class FederatedServerConfig:
    num_rounds: int = 3
    min_fit_clients: int = 2
    min_available_clients: int = 2
    min_evaluate_clients: int = 2
    strategy_kwargs: Mapping[str, Any] = field(default_factory=dict)


def start_task_server(
    task_name: str,
    server_address: str,
    config: FederatedServerConfig,
    on_fit_config_fn: Callable[[int], Mapping[str, Any]] | None = None,
    on_evaluate_config_fn: Callable[[int], Mapping[str, Any]] | None = None,
) -> None:
    strategy = TaskFedAvgStrategy(
        task_name=task_name,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        min_fit_clients=config.min_fit_clients,
        min_evaluate_clients=config.min_evaluate_clients,
        min_available_clients=config.min_available_clients,
        **dict(config.strategy_kwargs),
    )
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
    )


def start_rag_server(
    server_address: str,
    index: RagIndex,
    config: FederatedServerConfig,
    on_fit_config_fn: Callable[[int], Mapping[str, Any]] | None = None,
) -> None:
    strategy = RagFedAvgStrategy(
        rag_index=index,
        on_fit_config_fn=on_fit_config_fn,
        min_fit_clients=config.min_fit_clients,
        min_evaluate_clients=config.min_evaluate_clients,
        min_available_clients=config.min_available_clients,
        **dict(config.strategy_kwargs),
    )
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
    )


def build_task_client(task: TaskDefinition) -> FederatedTaskClient:
    return FederatedTaskClient(task=task)


def build_rag_client(
    index: RagIndex,
    document_provider: Callable[[Mapping[str, Any]], list[RagState]],
) -> RagClient:
    return RagClient(index=index, document_provider=document_provider)
