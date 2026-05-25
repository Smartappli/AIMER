# Copyright (c) 2026 AIMER contributors.
"""OpenRAG backend integration."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from packaging.version import InvalidVersion, Version

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - only used in minimal environments.
    class Document:  # type: ignore[no-redef]
        """Fallback Document type when langchain is unavailable."""

        def __init__(self, page_content: str, metadata: dict[str, Any] | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}


MIN_OPENRAG_VERSION = Version("0.4.1")


def _assert_supported_openrag_version() -> None:
    """Require OpenRAG runtime to be at least MIN_OPENRAG_VERSION."""
    try:
        installed = version("openrag")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "OpenRAG backend is required but dependency `openrag` is not installed.",
        ) from exc
    try:
        installed_version = Version(installed)
    except InvalidVersion as exc:
        raise RuntimeError(f"Unable to parse OpenRAG version string: {installed}.") from exc

    if installed_version < MIN_OPENRAG_VERSION:
        raise RuntimeError(
            f"OpenRAG>={MIN_OPENRAG_VERSION} "
            f"is required, found {installed}.",
        )


def _to_langchain_documents(results: list[Any]) -> list[Document]:
    """Normalize OpenRAG results into LangChain Documents."""
    normalized: list[Document] = []
    for item in results:
        if isinstance(item, Document):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            content = str(item.get("content", item.get("text", "")))
            metadata = item.get("metadata", {})
            normalized.append(Document(page_content=content, metadata=metadata))
            continue
        normalized.append(Document(page_content=str(item), metadata={}))
    return normalized


def _search_openrag(retriever: Any, query: str, k: int, filters: dict[str, Any]) -> list[Any]:
    """Call OpenRAG search with backward-compatible signatures."""
    attempts = (
        {"query": query, "top_k": k, "metadata_filters": filters},
        {"query": query, "k": k, "metadata_filters": filters},
        {"query": query, "top_k": k, "filters": filters},
        {"query": query, "k": k, "filters": filters},
    )
    for params in attempts:
        try:
            return retriever.search(**params)
        except TypeError:
            continue
    raise RuntimeError("OpenRAG retriever.search signature is incompatible with supported variants.")


def openrag_hybrid_search(
    query: str,
    k: int,
    filters: dict[str, Any] | None = None,
) -> list[Document]:
    """
    Query OpenRAG when available.

    Raises:
        RuntimeError: If OpenRAG dependency is not installed.
    """
    _assert_supported_openrag_version()
    try:
        from openrag import OpenRAG  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "OpenRAG backend is required but dependency `openrag` is not installed.",
        ) from exc

    retriever = OpenRAG(
        endpoint=os.getenv("OPENRAG_ENDPOINT", "http://localhost:8000"),
        api_key=os.getenv("OPENRAG_API_KEY"),
        collection=os.getenv("RAG_COLLECTION_NAME", "rag_docs"),
    )
    results = _search_openrag(retriever, query=query, k=k, filters=filters or {})
    return _to_langchain_documents(results)
