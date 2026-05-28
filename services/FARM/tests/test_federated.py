# Copyright (C) 2026 AIMER contributors.

"""Tests for FARM federated learning helpers."""

from __future__ import annotations

from typing import Any

import flwr as fl
import numpy as np
import pytest

from FARM.federated.clients import FederatedTaskClient, RagClient
from FARM.federated.data import FederatedDataset
from FARM.federated.rag import (
    RagDocument,
    RagIndex,
    RagState,
    parameters_to_state,
    state_to_parameters,
)
from FARM.federated.runner import (
    FederatedServerConfig,
    FederatedTransportSecurityError,
    _server_certificates,
    build_rag_client,
    build_task_client,
)
from FARM.federated.strategies import RagFedAvgStrategy, TaskFedAvgStrategy
from FARM.federated.tasks import (
    EvaluationResult,
    TaskDefinition,
    TaskType,
    TrainingResult,
)


class DummyHandlers:
    """Task handler implementation used by client delegation tests."""

    def __init__(self) -> None:
        self.model_parameters: list[np.ndarray] = [np.array([1.0])]
        self.received_parameters: list[np.ndarray] | None = None

    def get_parameters(self, _model: object) -> list[np.ndarray]:
        """Return deterministic model parameters."""
        return self.model_parameters

    def set_parameters(self, _model: object, parameters: list[np.ndarray]) -> None:
        """Capture incoming parameters."""
        self.received_parameters = parameters

    def train(
        self,
        _model: object,
        _dataset: FederatedDataset,
        config: dict[str, Any],
    ) -> TrainingResult:
        """Return a deterministic training result."""
        return TrainingResult(
            parameters=[np.array([2.0])],
            num_examples=5,
            metrics={"round": int(config["round"])},
        )

    def evaluate(
        self,
        _model: object,
        _dataset: FederatedDataset,
        _config: dict[str, Any],
    ) -> EvaluationResult:
        """Return a deterministic evaluation result."""
        return EvaluationResult(loss=0.25, num_examples=3, metrics={"ok": True})


def _rag_state(doc_id: str, values: list[float]) -> RagState:
    return RagState(
        doc_ids=[doc_id],
        embeddings=np.asarray([values], dtype=np.float32),
        texts=[f"text-{doc_id}"],
        metadata=[{"source": doc_id}],
    )


def test_rag_index_serializes_and_skips_duplicate_documents() -> None:
    """Ensure RAG index state round-trips through Flower parameters."""
    index = RagIndex(embedding_dim=2)

    index.add_documents(
        [
            RagDocument(
                doc_id="doc-1",
                embedding=[0.1, 0.2],
                text="first",
                metadata={"site": "a"},
            ),
            RagDocument(
                doc_id="doc-1",
                embedding=[0.9, 0.9],
                text="duplicate",
            ),
        ],
    )

    assert index.size == 1
    decoded = parameters_to_state(state_to_parameters(index.to_state()))
    assert decoded is not None
    assert decoded.doc_ids == ["doc-1"]
    assert decoded.texts == ["first"]
    assert decoded.embeddings.shape == (1, 2)


def test_rag_index_rejects_wrong_embedding_dimensions() -> None:
    """Ensure incompatible embeddings fail before mutating the index."""
    index = RagIndex(embedding_dim=2)

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        index.add_documents(
            [RagDocument(doc_id="bad", embedding=[1.0, 2.0, 3.0], text="bad")],
        )


def test_federated_task_client_delegates_to_handlers() -> None:
    """Ensure the task client delegates parameter, fit, and evaluation calls."""
    handlers = DummyHandlers()
    task = TaskDefinition(
        name="demo",
        task_type=TaskType.CLASSIFICATION,
        model=object(),
        dataset=FederatedDataset(train=[1, 2, 3]),
        handlers=handlers,
    )
    client = FederatedTaskClient(task)

    assert client.get_parameters() == handlers.model_parameters
    fit_parameters, fit_examples, fit_metrics = client.fit(
        [np.array([3.0])],
        {"round": 1},
    )
    loss, eval_examples, eval_metrics = client.evaluate([np.array([4.0])], {})

    assert handlers.received_parameters is not None
    assert fit_parameters[0].tolist() == [2.0]
    assert fit_examples == 5
    assert fit_metrics == {"round": 1}
    assert loss == 0.25
    assert eval_examples == 3
    assert eval_metrics == {"ok": True}


def test_rag_client_merges_incoming_and_local_updates() -> None:
    """Ensure RAG clients merge server state and local provider updates."""
    index = RagIndex(embedding_dim=2)

    client = RagClient(
        index=index,
        document_provider=lambda _config: [_rag_state("local", [0.3, 0.4])],
    )
    response = client.fit(
        fl.common.FitIns(
            parameters=state_to_parameters(_rag_state("remote", [0.1, 0.2])),
            config={},
        ),
    )
    decoded = parameters_to_state(response.parameters)

    assert response.num_examples == 2
    assert response.metrics == {"rag_docs": 2}
    assert decoded is not None
    assert decoded.doc_ids == ["remote", "local"]


def test_rag_client_returns_parameters_and_evaluation_metrics() -> None:
    """Ensure RAG clients expose Flower-compatible responses."""
    index = RagIndex(embedding_dim=2)
    index.merge_state(_rag_state("doc", [0.1, 0.2]))
    client = build_rag_client(index, document_provider=lambda _config: [])

    parameters = client.get_parameters(fl.common.GetParametersIns(config={}))
    evaluation = client.evaluate(
        fl.common.EvaluateIns(
            parameters=parameters.parameters,
            config={},
        ),
    )

    assert parameters.status.code == fl.common.Code.OK
    assert evaluation.loss == 0.0
    assert evaluation.metrics == {"rag_docs": 1}


def test_rag_strategy_aggregates_fit_results() -> None:
    """Ensure the RAG strategy merges client payloads into the shared index."""
    index = RagIndex(embedding_dim=2)
    strategy = RagFedAvgStrategy(rag_index=index)
    fit_result = fl.common.FitRes(
        status=fl.common.Status(code=fl.common.Code.OK, message="ok"),
        parameters=state_to_parameters(_rag_state("doc", [0.1, 0.2])),
        num_examples=1,
        metrics={},
    )

    initial = strategy.initialize_parameters(_client_manager=None)
    parameters, metrics = strategy.aggregate_fit(
        server_round=2,
        results=[(object(), fit_result)],
        _failures=None,
    )

    assert initial is not None
    assert parameters is not None
    assert metrics == {"rag_docs": 1, "round": 2}


def test_runner_factories_build_clients_and_config() -> None:
    """Ensure runner factories expose configured client objects."""
    handlers = DummyHandlers()
    task = TaskDefinition(
        name="demo",
        task_type=TaskType.OTHER,
        model=object(),
        dataset=FederatedDataset(train=[]),
        handlers=handlers,
    )
    config = FederatedServerConfig(strategy_kwargs={"fraction_fit": 1.0})
    strategy = TaskFedAvgStrategy(task_name="demo")

    assert isinstance(build_task_client(task), FederatedTaskClient)
    assert config.num_rounds == 3
    assert config.strategy_kwargs == {"fraction_fit": 1.0}
    assert strategy.task_name == "demo"


def test_federated_server_requires_tls_in_production(monkeypatch) -> None:
    """Ensure production Flower servers cannot start without TLS material."""
    monkeypatch.setenv("FARM_FEDERATED_ENVIRONMENT", "production")

    with pytest.raises(FederatedTransportSecurityError, match="TLS"):
        _server_certificates(FederatedServerConfig())


def test_federated_server_loads_tls_certificates(tmp_path) -> None:
    """Ensure configured Flower TLS files are loaded as byte certificates."""
    ca = tmp_path / "ca.pem"
    cert = tmp_path / "server.pem"
    key = tmp_path / "server.key"
    ca.write_bytes(b"ca")
    cert.write_bytes(b"cert")
    key.write_bytes(b"key")

    certificates = _server_certificates(
        FederatedServerConfig(
            require_tls=True,
            tls_ca_cert_path=str(ca),
            tls_server_cert_path=str(cert),
            tls_server_key_path=str(key),
        ),
    )

    assert certificates == (b"ca", b"cert", b"key")
