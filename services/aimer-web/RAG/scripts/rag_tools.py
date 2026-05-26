# Copyright (c) 2026 AIMER contributors.
"""RAG helper tools for metadata filtering and hybrid retrieval."""

from collections.abc import Sequence
from functools import lru_cache
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from RAG.openrag_backend import openrag_hybrid_search
from RAG.omop import build_omop_metadata
from RAG.scripts.schema import ChunkMetadata

load_dotenv()

# Configuration
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """Return the configured Ollama chat client lazily."""
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)


def extract_filters(user_query: str) -> dict[str, Any]:
    """
    Extract metadata filters from a user query using structured LLM output.

    Returns:
        dict[str, Any]: Non-null metadata fields inferred from the query.

    """
    prompt = f"""
            Extract metadata filters from the query.
            Return None for fields not mentioned.

                <USER QUERY STARTS>
                {user_query}
                <USER QUERY ENDS>

                Extract metadata based on the user query only:

            """

    structured_llm = get_llm().with_structured_output(ChunkMetadata)

    metadata = structured_llm.invoke(prompt)
    llm_filters = metadata.model_dump(exclude_none=True) if metadata else {}
    omop_filters = build_omop_metadata(user_query)
    for key, value in omop_filters.items():
        llm_filters.setdefault(key, value)

    return llm_filters


@tool
def hybrid_search(query: str, k: int = 5) -> Sequence[Document]:
    """
    Perform hybrid search (dense + sparse vector).

    Args:
        query: Search query.
        k: Number of results.

    Returns:
        Sequence[Document]: Matching documents.

    """
    filters = extract_filters(query)

    return openrag_hybrid_search(query=query, k=k, filters=filters)
