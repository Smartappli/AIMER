# Copyright (c) 2026 AIMER contributors.
"""Tests for strict OpenRAG retrieval mode in recommender."""

from RAG.recommender import OpenRAGRuntimeUnavailableError, recommend_models_for_query


def test_recommender_strict_mode_raises_when_retrieval_fallback(monkeypatch) -> None:
    def fake_retrieve(query: str, *, k: int):
        del query, k
        return [], {}, "catalog-only-fallback"

    monkeypatch.setattr("RAG.recommender._safe_retrieve", fake_retrieve)

    try:
        recommend_models_for_query(
            query="classification chest xray",
            strict_openrag=True,
        )
    except OpenRAGRuntimeUnavailableError as exc:
        assert "OpenRAG retrieval is required" in str(exc)
    else:
        raise AssertionError(
            "Expected OpenRAGRuntimeUnavailableError in strict OpenRAG mode"
        )


def test_resolve_strict_openrag_uses_env_default_true(monkeypatch) -> None:
    from RAG.recommender import _resolve_strict_openrag

    monkeypatch.delenv("RAG_STRICT_OPENRAG", raising=False)
    assert _resolve_strict_openrag(None) is True


def test_resolve_strict_openrag_allows_false_values(monkeypatch) -> None:
    from RAG.recommender import _resolve_strict_openrag

    monkeypatch.setenv("RAG_STRICT_OPENRAG", "false")
    assert _resolve_strict_openrag(None) is False


def test_resolve_strict_openrag_explicit_argument_wins(monkeypatch) -> None:
    from RAG.recommender import _resolve_strict_openrag

    monkeypatch.setenv("RAG_STRICT_OPENRAG", "0")
    assert _resolve_strict_openrag(True) is True


def test_resolve_strict_openrag_forces_true_in_production(monkeypatch) -> None:
    from RAG.recommender import _resolve_strict_openrag

    monkeypatch.setenv("AIMER_RAG_ENVIRONMENT", "production")
    monkeypatch.setenv("RAG_STRICT_OPENRAG", "false")
    assert _resolve_strict_openrag(False) is True


def test_recommender_does_not_return_ungrounded_recommendations() -> None:
    response = recommend_models_for_query(
        query="classification chest xray",
        documents=[],
        strict_openrag=False,
    )

    assert response.recommended_models == []
    assert response.no_recommendation_reason is not None


def test_recommender_rejects_ungrounded_recommendations_in_production(
    monkeypatch,
) -> None:
    from RAG.recommender import _resolve_allow_ungrounded

    monkeypatch.setenv("AIMER_RAG_ENVIRONMENT", "production")
    try:
        _resolve_allow_ungrounded(True)
    except RuntimeError as exc:
        assert "Ungrounded recommendations" in str(exc)
    else:
        raise AssertionError("Expected production to reject ungrounded output")
