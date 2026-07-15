# Copyright (c) 2026 AIMER contributors.
"""Client boundary between Django and the RAG recommendation service."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import httpx
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


def _rag_service_api_key() -> str:
    """Return the optional service-to-service API key."""
    return str(getattr(settings, "RAG_SERVICE_API_KEY", "") or "").strip()


def _rag_service_ca_cert_path() -> str:
    """Return the optional CA bundle used to verify the RAG service."""
    return str(getattr(settings, "RAG_SERVICE_CA_CERT_PATH", "") or "").strip()


def _decode_json_response(response: httpx.Response) -> dict[str, Any]:
    """Decode an HTTP response body as a JSON object."""
    payload = response.json()
    if not isinstance(payload, dict):
        msg = "RAG service returned a non-object JSON payload."
        raise RagServiceUnavailableError(msg)
    return payload


def _error_detail(response: httpx.Response) -> str:
    """Extract a concise error detail from a failed service response."""
    try:
        payload = response.json()
    except ValueError:
        return response.text or response.reason_phrase
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error")
        if isinstance(detail, str):
            return detail
    return response.reason_phrase


def _remote_json_request(
    path: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Call the configured RAG service and return its JSON object payload."""
    base_url = _rag_service_url()
    if not base_url:
        msg = "RAG_SERVICE_URL is not configured."
        raise RagServiceUnavailableError(msg)
    parsed_url = urlsplit(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        msg = "RAG_SERVICE_URL must be an HTTP(S) URL."
        raise RagServiceUnavailableError(msg)
    if getattr(settings, "IS_PRODUCTION", False) and parsed_url.scheme != "https":
        msg = "RAG_SERVICE_URL must use HTTPS in production."
        raise RagServiceUnavailableError(msg)

    headers = {"Accept": "application/json"}
    api_key = _rag_service_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    method = "GET"
    json_payload = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        method = "POST"
        json_payload = payload

    try:
        request_kwargs = {
            "headers": headers,
            "json": json_payload,
            "timeout": _rag_service_timeout(),
        }
        ca_cert_path = _rag_service_ca_cert_path()
        if ca_cert_path:
            with httpx.Client(verify=ca_cert_path) as client:
                response = client.request(method, f"{base_url}{path}", **request_kwargs)
        else:
            response = httpx.request(method, f"{base_url}{path}", **request_kwargs)
        if response.is_error:
            raise RagServiceUnavailableError(_error_detail(response))
        return _decode_json_response(response)
    except (httpx.RequestError, ValueError) as exc:
        raise RagServiceUnavailableError(str(exc)) from exc


def recommend_models(
    *,
    query: str,
    top_k: int,
    strict_openrag: bool = True,
    language: str = "fr",
) -> dict[str, Any]:
    """Return recommendation payloads through the remote service or local fallback."""
    if _rag_service_url():
        return _remote_json_request(
            "/recommend",
            {
                "query": query,
                "top_k": top_k,
                "strict_openrag": strict_openrag,
                "language": language,
            },
        )

    from RAG.recommender import (
        OpenRAGRuntimeUnavailableError,
        recommend_models_for_query,
    )

    try:
        payload = recommend_models_for_query(
            query=query,
            top_k=top_k,
            strict_openrag=strict_openrag,
            language=language,
        )
    except OpenRAGRuntimeUnavailableError as exc:
        raise RagServiceUnavailableError(str(exc)) from exc
    return payload.model_dump()


def runtime_status() -> dict[str, Any]:
    """Return readiness payloads through the remote service or local fallback."""
    if _rag_service_url():
        return _remote_json_request("/readyz")

    from RAG.healthcheck import is_rag_runtime_ready, rag_runtime_health

    return {"ready": is_rag_runtime_ready(), "status": rag_runtime_health()}
