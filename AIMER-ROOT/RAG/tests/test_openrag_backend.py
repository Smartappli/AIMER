# Copyright (c) 2026 AIMER contributors.
"""Tests for mandatory OpenRAG backend behavior."""

from importlib.metadata import PackageNotFoundError

from RAG.openrag_backend import Document
from RAG.openrag_backend import MIN_OPENRAG_VERSION
from RAG.openrag_backend import _search_openrag
from RAG.openrag_backend import _to_langchain_documents
from RAG.openrag_backend import openrag_hybrid_search


def test_openrag_hybrid_search_raises_when_dependency_missing(monkeypatch) -> None:
    def fake_version(_: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("RAG.openrag_backend.version", fake_version)

    try:
        openrag_hybrid_search(query="test", k=3, filters={"doc_year": 2026})
    except RuntimeError as exc:
        assert "required" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when openrag is missing")


def test_to_langchain_documents_normalizes_dict_and_text() -> None:
    docs = _to_langchain_documents(
        [{"content": "alpha", "metadata": {"a": 1}}, {"text": "beta"}, "gamma"],
    )

    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "alpha"
    assert docs[0].metadata == {"a": 1}
    assert docs[1].page_content == "beta"
    assert docs[2].page_content == "gamma"


def test_minimum_openrag_version_is_pinned() -> None:
    assert str(MIN_OPENRAG_VERSION) == "0.4.1"


def test_search_openrag_supports_top_k_metadata_filters_signature() -> None:
    class DummyRetriever:
        def search(self, **kwargs):
            assert kwargs["top_k"] == 5
            assert kwargs["metadata_filters"] == {"doc_year": 2026}
            return [{"content": "ok"}]

    result = _search_openrag(
        DummyRetriever(),
        query="test",
        k=5,
        filters={"doc_year": 2026},
    )
    assert result == [{"content": "ok"}]
