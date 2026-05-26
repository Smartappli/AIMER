# Copyright (c) 2026 AIMER contributors.
"""Runtime health checks for the OpenRAG-based RAG stack."""

from __future__ import annotations

from importlib.util import find_spec
import os
from urllib.parse import urlsplit


def _openrag_endpoint_is_valid() -> bool:
    """Return whether OPENRAG_ENDPOINT is a usable HTTP(S) URL."""
    endpoint = os.getenv("OPENRAG_ENDPOINT", "").strip()
    parsed = urlsplit(endpoint)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def rag_runtime_health() -> dict[str, bool]:
    """Return dependency and configuration readiness flags."""
    return {
        "openrag_installed": find_spec("openrag") is not None,
        "langchain_ollama_installed": find_spec("langchain_ollama") is not None,
        "langchain_core_installed": find_spec("langchain_core") is not None,
        "dotenv_installed": find_spec("dotenv") is not None,
        "openrag_endpoint_set": bool(os.getenv("OPENRAG_ENDPOINT")),
        "openrag_endpoint_valid": _openrag_endpoint_is_valid(),
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
        and status["openrag_endpoint_valid"]
    )
    return bool(required)
