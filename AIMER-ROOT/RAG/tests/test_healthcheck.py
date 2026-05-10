# Copyright (c) 2026 AIMER contributors.
"""Tests for RAG runtime health checks."""

from RAG.healthcheck import is_rag_runtime_ready
from RAG.healthcheck import rag_runtime_health


def test_rag_runtime_health_shape() -> None:
    status = rag_runtime_health()
    expected = {
        "openrag_installed",
        "langchain_ollama_installed",
        "langchain_core_installed",
        "dotenv_installed",
        "openrag_endpoint_set",
    }
    assert set(status.keys()) == expected
    assert all(isinstance(value, bool) for value in status.values())


def test_is_rag_runtime_ready_returns_bool() -> None:
    assert isinstance(is_rag_runtime_ready(), bool)


def test_rag_runtime_not_ready_without_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("OPENRAG_ENDPOINT", raising=False)
    assert is_rag_runtime_ready() is False


def test_verify_openrag_report_mentions_runtime_ready() -> None:
    from RAG.verify_openrag import format_report

    report = format_report()
    assert "OpenRAG runtime verification:" in report
    assert "runtime_ready:" in report


def test_verify_openrag_main_returns_status_code(monkeypatch) -> None:
    from RAG.verify_openrag import main

    monkeypatch.setenv("OPENRAG_ENDPOINT", "http://localhost:8000")
    status = main()
    assert status in (0, 1)


def test_verify_openrag_main_returns_one_when_endpoint_missing(monkeypatch) -> None:
    from RAG.verify_openrag import main

    monkeypatch.delenv("OPENRAG_ENDPOINT", raising=False)
    assert main() == 1
