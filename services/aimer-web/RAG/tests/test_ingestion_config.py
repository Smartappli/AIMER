# Copyright (c) 2026 AIMER contributors.
"""Tests for RAG ingestion runtime configuration."""

from __future__ import annotations

import importlib


def test_ingestion_config_uses_environment(monkeypatch) -> None:
    """Ensure ingestion endpoints and secrets are read from environment."""
    ingestion = importlib.import_module("RAG.main")
    with monkeypatch.context() as patched:
        patched.setenv("QDRANT_URL", "http://qdrant:6333")
        patched.setenv("QDRANT_API_KEY", "test-secret")
        patched.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
        patched.setenv("RAG_VISION_MODEL", "vision-model")
        patched.setenv("RAG_EMBEDDING_MODEL", "embedding-model")

        ingestion = importlib.reload(ingestion)

        assert ingestion.QDRANT_URL == "http://qdrant:6333"
        assert ingestion.QDRANT_API_KEY == "test-secret"
        assert ingestion.OLLAMA_BASE_URL == "http://ollama:11434"
        assert ingestion.MODEL_NAME == "vision-model"
        assert ingestion.EMBEDDING_MODEL == "embedding-model"

    importlib.reload(ingestion)


def test_ingestion_disables_external_plugins_by_default(monkeypatch) -> None:
    """Ensure Docling external plugins require explicit opt-in."""
    ingestion = importlib.import_module("RAG.main")
    with monkeypatch.context() as patched:
        patched.delenv("RAG_ALLOW_EXTERNAL_PLUGINS", raising=False)
        ingestion = importlib.reload(ingestion)

        assert ingestion.ALLOW_EXTERNAL_PLUGINS is False

    importlib.reload(ingestion)


def test_ingestion_rejects_external_plugins_in_production(monkeypatch) -> None:
    """Ensure regulated ingestion cannot enable external plugins."""
    ingestion = importlib.import_module("RAG.main")
    with monkeypatch.context() as patched:
        patched.setenv("AIMER_RAG_ENVIRONMENT", "production")
        patched.setenv("QDRANT_API_KEY", "qdrant-secret")
        patched.setenv("RAG_ALLOW_EXTERNAL_PLUGINS", "true")
        ingestion = importlib.reload(ingestion)

        try:
            ingestion.validate_ingestion_configuration()
        except RuntimeError as exc:
            assert "RAG_ALLOW_EXTERNAL_PLUGINS" in str(exc)
        else:
            raise AssertionError("Expected production external plugins to fail")

    importlib.reload(ingestion)


def test_ingestion_requires_qdrant_api_key_in_production(monkeypatch) -> None:
    """Ensure production ingestion cannot write to unauthenticated Qdrant."""
    ingestion = importlib.import_module("RAG.main")
    with monkeypatch.context() as patched:
        patched.setenv("AIMER_RAG_ENVIRONMENT", "production")
        patched.delenv("QDRANT_API_KEY", raising=False)
        patched.delenv("RAG_ALLOW_EXTERNAL_PLUGINS", raising=False)
        ingestion = importlib.reload(ingestion)

        try:
            ingestion.validate_ingestion_configuration()
        except RuntimeError as exc:
            assert "QDRANT_API_KEY" in str(exc)
        else:
            raise AssertionError("Expected production Qdrant key to fail")

    importlib.reload(ingestion)
