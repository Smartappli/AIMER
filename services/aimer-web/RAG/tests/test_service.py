# Copyright (c) 2026 AIMER contributors.
"""Tests for the standalone RAG HTTP API."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from RAG.recommender import (
    OpenRAGRuntimeUnavailableError,
    QueryProfile,
    RecommendationResponse,
)
from RAG.service import app


def _recommendation_response(query: str) -> RecommendationResponse:
    """Build a minimal valid recommendation response."""
    return RecommendationResponse(
        query=query,
        query_profile=QueryProfile(
            tasks=[],
            modalities=[],
            query_tokens=[],
            omop_condition_concept_ids=[],
            omop_modality_concept_ids=[],
            snomed_ct_codes=[],
        ),
        used_filters={},
        retrieval_mode="hybrid+rerank",
        safety_notice="test",
        recommended_models=[],
    )


def test_healthz_returns_liveness_payload() -> None:
    """Ensure deployment smoke tests have a stable liveness endpoint."""
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"service": "aimer-rag", "status": "ok"}


@patch("RAG.service.recommend_models_for_query")
def test_recommend_endpoint_returns_recommendation_payload(mock_recommend) -> None:
    """Ensure the HTTP endpoint forwards request fields to the engine."""
    mock_recommend.return_value = _recommendation_response("classification mri")
    client = TestClient(app)

    response = client.post(
        "/recommend",
        json={
            "query": "classification mri",
            "top_k": 2,
            "strict_openrag": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["query"] == "classification mri"
    mock_recommend.assert_called_once_with(
        query="classification mri",
        top_k=2,
        strict_openrag=True,
    )


@patch("RAG.service.recommend_models_for_query")
def test_recommend_endpoint_returns_503_when_runtime_unavailable(
    mock_recommend,
) -> None:
    """Ensure OpenRAG runtime failures are surfaced as service-unavailable."""
    mock_recommend.side_effect = OpenRAGRuntimeUnavailableError("OpenRAG unavailable")
    client = TestClient(app)

    response = client.post(
        "/recommend",
        json={"query": "classification mri", "top_k": 2},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "OpenRAG unavailable"
