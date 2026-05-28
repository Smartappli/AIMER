# Copyright (c) 2026 AIMER contributors.
"""Tests for the standalone RAG HTTP API."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from RAG.recommender import (
    OpenRAGRuntimeUnavailableError,
    QueryProfile,
    RecommendationResponse,
)
from RAG.service import app, validate_service_configuration


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


def test_production_requires_rag_service_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure production cannot silently expose RAG service routes."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("AIMER_RAG_ENVIRONMENT", raising=False)
    monkeypatch.delenv("AIMER_RAG_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="AIMER_RAG_API_KEY"):
        validate_service_configuration()


def test_production_rejects_development_rag_service_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure placeholder service keys cannot satisfy production validation."""
    monkeypatch.setenv("AIMER_RAG_ENVIRONMENT", "production")
    monkeypatch.setenv("AIMER_RAG_API_KEY", "dev-insecure-rag-key-change-me")

    with pytest.raises(RuntimeError, match="development/test prefix"):
        validate_service_configuration()


def test_production_rejects_ungrounded_recommendation_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure production startup rejects catalog-only recommendation opt-in."""
    monkeypatch.setenv("AIMER_RAG_ENVIRONMENT", "production")
    monkeypatch.setenv("AIMER_RAG_API_KEY", "service-secret")
    monkeypatch.setenv("RAG_ALLOW_UNGROUNDED_RECOMMENDATIONS", "true")

    with pytest.raises(RuntimeError, match="UNGROUNDED"):
        validate_service_configuration()


@patch("RAG.service.recommend_models_for_query")
def test_recommend_endpoint_requires_api_key_when_configured(
    mock_recommend: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure service API auth is enforced when configured."""
    monkeypatch.setenv("AIMER_RAG_API_KEY", "service-secret")
    mock_recommend.return_value = _recommendation_response("classification mri")
    client = TestClient(app)

    unauthorized = client.post("/recommend", json={"query": "classification mri"})
    authorized = client.post(
        "/recommend",
        headers={"Authorization": "Bearer service-secret"},
        json={"query": "classification mri"},
    )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


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
def test_recommend_endpoint_forces_strict_openrag_in_production(
    mock_recommend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure production requests cannot disable strict retrieval."""
    monkeypatch.setenv("AIMER_RAG_ENVIRONMENT", "production")
    monkeypatch.setenv("AIMER_RAG_API_KEY", "service-secret")
    mock_recommend.return_value = _recommendation_response("classification mri")
    client = TestClient(app)

    response = client.post(
        "/recommend",
        headers={"X-API-Key": "service-secret"},
        json={
            "query": "classification mri",
            "strict_openrag": False,
        },
    )

    assert response.status_code == 200
    mock_recommend.assert_called_once_with(
        query="classification mri",
        top_k=3,
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
