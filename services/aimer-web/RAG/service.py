# Copyright (c) 2026 AIMER contributors.
"""HTTP API boundary for the RAG recommendation runtime."""

from __future__ import annotations

import hmac
import os
from collections.abc import Awaitable, Callable
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from RAG.healthcheck import is_rag_runtime_ready, rag_runtime_health
from RAG.recommender import (
    OpenRAGRuntimeUnavailableError,
    RecommendationResponse,
    recommend_models_for_query,
)

MAX_TOP_K = 10
DEFAULT_MAX_QUERY_LENGTH = 2000


def _max_query_length() -> int:
    """Return a bounded maximum size for user-provided RAG queries."""
    raw_value = os.getenv(
        "RAG_RECOMMENDATION_MAX_QUERY_LENGTH",
        str(DEFAULT_MAX_QUERY_LENGTH),
    )
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_MAX_QUERY_LENGTH


MAX_QUERY_LENGTH = _max_query_length()


class RecommendationRequest(BaseModel):
    """Request payload accepted by the RAG recommendation API."""

    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    top_k: int = Field(default=3, ge=1, le=MAX_TOP_K)
    strict_openrag: bool = True
    language: Literal["fr", "en", "nl", "de"] = "fr"


app = FastAPI(title="AIMER RAG Service")

PUBLIC_PATHS = {"/healthz", "/readyz"}


def _is_production() -> bool:
    """Return whether this service is running in a production environment."""
    environment = os.getenv(
        "AIMER_RAG_ENVIRONMENT",
        os.getenv("ENVIRONMENT", "local"),
    )
    return environment.strip().lower() in {"prod", "production"}


def _configured_api_key() -> str:
    """Return the configured service API key, if any."""
    return os.getenv("AIMER_RAG_API_KEY", "").strip()


def validate_service_configuration() -> None:
    """Fail fast when regulated production service auth is not configured."""
    if not _is_production():
        return

    api_key = _configured_api_key()
    if not api_key:
        msg = "AIMER_RAG_API_KEY must be set when ENVIRONMENT=production."
        raise RuntimeError(msg)
    if api_key.startswith(("dev-", "test-", "ci-")):
        msg = "AIMER_RAG_API_KEY must not use a development/test prefix."
        raise RuntimeError(msg)
    if os.getenv("RAG_ALLOW_UNGROUNDED_RECOMMENDATIONS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        msg = "RAG_ALLOW_UNGROUNDED_RECOMMENDATIONS must be false in production."
        raise RuntimeError(msg)


def _request_api_key(request: Request) -> str:
    """Extract API key from Authorization bearer or X-API-Key headers."""
    authorization = request.headers.get("authorization", "")
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() == "bearer" and value:
        return value.strip()
    return request.headers.get("x-api-key", "").strip()


@app.middleware("http")
async def require_service_api_key(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Require a service key when AIMER_RAG_API_KEY is configured."""
    expected_key = _configured_api_key()
    if not expected_key or request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    provided_key = _request_api_key(request)
    if not provided_key or not hmac.compare_digest(provided_key, expected_key):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Return a stable liveness payload for deployment smoke tests."""
    return {"service": "aimer-rag", "status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, object]:
    """Return runtime readiness flags for OpenRAG-backed retrieval."""
    status = rag_runtime_health()
    return {"ready": is_rag_runtime_ready(), "status": status}


@app.post(
    "/recommend",
    responses={503: {"description": "RAG runtime unavailable"}},
)
async def recommend(payload: RecommendationRequest) -> RecommendationResponse:
    """Return model recommendations from the RAG corpus."""
    try:
        return recommend_models_for_query(
            query=payload.query.strip(),
            top_k=payload.top_k,
            strict_openrag=True if _is_production() else payload.strict_openrag,
            language=payload.language,
        )
    except OpenRAGRuntimeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


validate_service_configuration()
