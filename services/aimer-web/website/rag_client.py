# Copyright (c) 2026 AIMER contributors.
"""Client boundary between Django and the RAG recommendation service."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from django.conf import settings

DEFAULT_TIMEOUT_SECONDS = 5.0


class RagServiceUnavailableError(RuntimeError):
    """Raised when the RAG service or runtime cannot satisfy a request."""


def _rag_service_url() -> str:
    """Return the configured RAG service base URL, without a trailing slash."""
    return str(getattr(settings, "RAG_SERVICE_URL", "") or "").rstrip("/")


def _rag_service_timeout() -> float:
    """Return the configured RAG service timeout in seconds."""
    return float(
        getattr(
            settings,
            "RAG_SERVICE_TIMEOUT_SECONDS",
            DEFAULT_TIMEOUT_SECONDS,
        ),
    )


def _decode_json_response(response: object) -> dict[str, Any]:
    """Decode a urllib HTTP response body as JSON."""
    body = response.read().decode("utf-8")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        msg = "RAG service returned a non-object JSON payload."
        raise RagServiceUnavailableError(msg)
    return payload


def _error_detail(exc: HTTPError) -> str:
    """Extract a concise error detail from a failed service response."""
    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return str(exc)
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error")
        if isinstance(detail, str):
            return detail
    return str(exc)


def _remote_json_request(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call the configured RAG service and return its JSON object payload."""
    base_url = _rag_service_url()
    if not base_url:
        msg = "RAG_SERVICE_URL is not configured."
        raise RagServiceUnavailableError(msg)

    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"

    request = Request(
        f"{base_url}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=_rag_service_timeout()) as response:
            return _decode_json_response(response)
    except HTTPError as exc:
        raise RagServiceUnavailableError(_error_detail(exc)) from exc
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RagServiceUnavailableError(str(exc)) from exc


def recommend_models(
    *,
    query: str,
    top_k: int,
    strict_openrag: bool = True,
) -> dict[str, Any]:
    """Return recommendation payloads through the remote service or local fallback."""
    if _rag_service_url():
        return _remote_json_request(
            "/recommend",
            {
                "query": query,
                "top_k": top_k,
                "strict_openrag": strict_openrag,
            },
        )

    from RAG.recommender import (  # noqa: PLC0415
        OpenRAGRuntimeUnavailableError,
        recommend_models_for_query,
    )

    try:
        payload = recommend_models_for_query(
            query=query,
            top_k=top_k,
            strict_openrag=strict_openrag,
        )
    except OpenRAGRuntimeUnavailableError as exc:
        raise RagServiceUnavailableError(str(exc)) from exc
    return payload.model_dump()


def runtime_status() -> dict[str, Any]:
    """Return readiness payloads through the remote service or local fallback."""
    if _rag_service_url():
        return _remote_json_request("/readyz")

    from RAG.healthcheck import is_rag_runtime_ready, rag_runtime_health  # noqa: PLC0415

    return {"ready": is_rag_runtime_ready(), "status": rag_runtime_health()}
