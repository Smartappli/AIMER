# Copyright (c) 2026 AIMER contributors.
"""Runtime health checks for the OpenRAG-based RAG stack."""

from __future__ import annotations

from importlib.util import find_spec
import os


def rag_runtime_health() -> dict[str, bool]:
    """Return dependency and configuration readiness flags."""
    return {
        "openrag_installed": find_spec("openrag") is not None,
        "langchain_ollama_installed": find_spec("langchain_ollama") is not None,
        "langchain_core_installed": find_spec("langchain_core") is not None,
        "dotenv_installed": find_spec("dotenv") is not None,
        "openrag_endpoint_set": bool(os.getenv("OPENRAG_ENDPOINT")),
    }


def is_rag_runtime_ready() -> bool:
    """Whether required runtime dependencies/config are present."""
    status = rag_runtime_health()
    required = (
        status["openrag_installed"]
        and status["langchain_ollama_installed"]
        and status["langchain_core_installed"]
        and status["dotenv_installed"]
        and status["openrag_endpoint_set"]
    )
    return bool(required)
