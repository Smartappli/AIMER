# Copyright (c) 2026 AIMER contributors.
"""HTTP API boundary for the RAG recommendation runtime."""

from __future__ import annotations

import hmac
import os
from collections.abc import Awaitable, Callable

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


class RecommendationRequest(BaseModel):
    """Request payload accepted by the RAG recommendation API."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=MAX_TOP_K)
    strict_openrag: bool = True


app = FastAPI(title="AIMER RAG Service")

PUBLIC_PATHS = {"/healthz"}


def _configured_api_key() -> str:
    """Return the configured service API key, if any."""
    return os.getenv("AIMER_RAG_API_KEY", "").strip()


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
            strict_openrag=payload.strict_openrag,
        )
    except OpenRAGRuntimeUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
