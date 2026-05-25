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
