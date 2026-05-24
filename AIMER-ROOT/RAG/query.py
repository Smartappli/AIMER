# Copyright (c) 2026 AIMER contributors.
"""Query and reranking helpers for the OpenRAG backend."""

from __future__ import annotations

import operator
import os
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_ollama import ChatOllama

from RAG.openrag_backend import openrag_hybrid_search
from RAG.omop import build_omop_metadata
from RAG.scripts.schema import ChunkMetadata

if TYPE_CHECKING:
    from langchain_core.documents import Document

load_dotenv()

LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen3:8b")
RERANKER_MODEL = "Krakekai/qwen3-reranker-8b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)


def extract_filters(user_query: str) -> dict[str, Any]:
    """
    Extract metadata filters from a user query.

    Returns:
        Mapping of metadata keys and values inferred from the query.

    """
    prompt = f"""
            Extract metadata filters from the query. Return None for fields not mentioned.

            <USER QUERY STARTS>
            {user_query}
            </USER QUERY ENDS>
        """
    structured_llm = llm.with_structured_output(ChunkMetadata)
    metadata = structured_llm.invoke(prompt)
    llm_filters = metadata.model_dump(exclude_none=True) if metadata else {}
    omop_filters = build_omop_metadata(user_query)
    for key, value in omop_filters.items():
        llm_filters.setdefault(key, value)
    return llm_filters


def hybrid_search(
    query: str,
    k: int = 10,
    filters: dict[str, Any] | None = None,
) -> list[Document]:
    """
    Perform hybrid retrieval via OpenRAG.

    Returns:
        List of retrieved documents matching the query and optional filters.

    """
    return openrag_hybrid_search(query=query, k=k, filters=filters)


def rerank_results(
    query: str,
    documents: list[Document],
    top_k: int = 5,
) -> list[Document]:
    """
    Rerank retrieved documents using a cross-encoder.

    Returns:
        Top-ranked documents ordered by reranker relevance score.

    """
    reranker = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL,
        model_kwargs={"device": "xpu"},
    )
    query_doc_pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.score(query_doc_pairs)
    reranked = sorted(
        zip(scores, documents, strict=False),
        key=operator.itemgetter(0),
        reverse=True,
    )[:top_k]
    return [rank[1] for rank in reranked]
