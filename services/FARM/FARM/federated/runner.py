# Copyright (c) 2026 AIMER contributors.

"""Server and client factory helpers for federated execution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import flwr as fl

from .clients import FederatedTaskClient, RagClient
from .strategies import RagFedAvgStrategy, TaskFedAvgStrategy

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .rag import RagIndex, RagState
    from .tasks import TaskDefinition


ServerCertificates = tuple[bytes, bytes, bytes]


class FederatedTransportSecurityError(RuntimeError):
    """Raised when federated transport security is not production-ready."""


@dataclass(slots=True)
class FederatedServerConfig:
    """Configuration values used when starting a Flower server."""

    num_rounds: int = 3
    min_fit_clients: int = 2
    min_available_clients: int = 2
    min_evaluate_clients: int = 2
    strategy_kwargs: Mapping[str, Any] = field(default_factory=dict)
    require_tls: bool | None = None
    tls_ca_cert_path: str | None = None
    tls_server_cert_path: str | None = None
    tls_server_key_path: str | None = None


def _is_production() -> bool:
    """Return whether FARM federated runtime is in production mode."""
    environment = os.getenv(
        "FARM_FEDERATED_ENVIRONMENT",
        os.getenv("DJANGO_ENVIRONMENT", os.getenv("ENVIRONMENT", "local")),
    )
    return environment.strip().lower() in {"prod", "production"}


def _requires_tls(config: FederatedServerConfig) -> bool:
    """Resolve whether TLS certificates are mandatory."""
    if _is_production():
        return True
    return bool(config.require_tls)


def _read_certificate(path_value: str | None, label: str) -> bytes:
    """Read a configured certificate file."""
    if not path_value:
        msg = f"{label} is required when Flower TLS is enabled."
        raise FederatedTransportSecurityError(msg)
    path = Path(path_value)
    if not path.is_file():
        msg = f"{label} does not exist: {path}"
        raise FederatedTransportSecurityError(msg)
    return path.read_bytes()


def _server_certificates(config: FederatedServerConfig) -> ServerCertificates | None:
    """Load Flower server certificates or enforce TLS in production."""
    paths = (
        config.tls_ca_cert_path,
        config.tls_server_cert_path,
        config.tls_server_key_path,
    )
    if not any(paths):
        if _requires_tls(config):
            msg = (
                "Flower TLS certificates are required in production. Set "
                "tls_ca_cert_path, tls_server_cert_path and tls_server_key_path."
            )
            raise FederatedTransportSecurityError(msg)
        return None
    if not all(paths):
        msg = "All Flower TLS certificate paths must be set together."
        raise FederatedTransportSecurityError(msg)
    return (
        _read_certificate(config.tls_ca_cert_path, "tls_ca_cert_path"),
        _read_certificate(config.tls_server_cert_path, "tls_server_cert_path"),
        _read_certificate(config.tls_server_key_path, "tls_server_key_path"),
    )


def start_task_server(
    task_name: str,
    server_address: str,
    config: FederatedServerConfig,
    on_fit_config_fn: Callable[[int], Mapping[str, Any]] | None = None,
    on_evaluate_config_fn: Callable[[int], Mapping[str, Any]] | None = None,
) -> None:
    """Start a Flower server using task-oriented FedAvg strategy."""
    certificates = _server_certificates(config)
    strategy = TaskFedAvgStrategy(
        task_name=task_name,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        min_fit_clients=config.min_fit_clients,
        min_evaluate_clients=config.min_evaluate_clients,
        min_available_clients=config.min_available_clients,
        **dict(config.strategy_kwargs),
    )
    start_kwargs: dict[str, Any] = {
        "server_address": server_address,
        "config": fl.server.ServerConfig(num_rounds=config.num_rounds),
        "strategy": strategy,
    }
    if certificates is not None:
        start_kwargs["certificates"] = certificates
    fl.server.start_server(**start_kwargs)


def start_rag_server(
    server_address: str,
    index: RagIndex,
    config: FederatedServerConfig,
    on_fit_config_fn: Callable[[int], Mapping[str, Any]] | None = None,
) -> None:
    """Start a Flower server that aggregates shared RAG documents."""
    certificates = _server_certificates(config)
    strategy = RagFedAvgStrategy(
        rag_index=index,
        on_fit_config_fn=on_fit_config_fn,
        min_fit_clients=config.min_fit_clients,
        min_evaluate_clients=config.min_evaluate_clients,
        min_available_clients=config.min_available_clients,
        **dict(config.strategy_kwargs),
    )
    start_kwargs: dict[str, Any] = {
        "server_address": server_address,
        "config": fl.server.ServerConfig(num_rounds=config.num_rounds),
        "strategy": strategy,
    }
    if certificates is not None:
        start_kwargs["certificates"] = certificates
    fl.server.start_server(**start_kwargs)


def build_task_client(task: TaskDefinition) -> FederatedTaskClient:
    """
    Build a task client from a task definition.

    Returns:
        Initialized federated task client.

    """
    return FederatedTaskClient(task=task)


def build_rag_client(
    index: RagIndex,
    document_provider: Callable[[Mapping[str, Any]], list[RagState]],
) -> RagClient:
    """
    Build a RAG client from an index and update provider.

    Returns:
        Initialized RAG client bound to ``index`` and ``document_provider``.

    """
    return RagClient(index=index, document_provider=document_provider)
