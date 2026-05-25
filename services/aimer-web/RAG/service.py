# Copyright (c) 2026 AIMER contributors.
"""HTTP API boundary for the RAG recommendation runtime."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
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


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Return a stable liveness payload for deployment smoke tests."""
    return {"service": "aimer-rag", "status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, object]:
    """Return runtime readiness flags for OpenRAG-backed retrieval."""
    status = rag_runtime_health()
    return {"ready": is_rag_runtime_ready(), "status": status}


@app.post("/recommend", response_model=RecommendationResponse)
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
