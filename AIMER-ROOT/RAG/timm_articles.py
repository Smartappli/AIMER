# Copyright (c) 2026 AIMER contributors.
"""Utilities to expose the default TIMM paper index for the RAG knowledge base."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class TimmArticle(TypedDict):
    """One TIMM model-to-paper mapping entry used for KB bootstrap."""

    model_name: str
    paper_title: str
    paper_url: str


def timm_article_index_file() -> Path:
    """Return the versioned JSON file storing default TIMM paper references."""
    return Path(__file__).resolve().parent / "data" / "timm_model_articles.json"


def load_timm_article_index() -> list[TimmArticle]:
    """
    Load the default TIMM paper references bundled with the repository.

    Returns:
        List of TIMM paper references. Empty list when the file is missing or invalid.

    """
    path = timm_article_index_file()
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    cleaned: list[TimmArticle] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_name = item.get("model_name")
        paper_title = item.get("paper_title")
        paper_url = item.get("paper_url")
        if (
            isinstance(model_name, str)
            and isinstance(paper_title, str)
            and isinstance(paper_url, str)
            and model_name.strip()
            and paper_title.strip()
            and paper_url.strip()
        ):
            cleaned.append(
                TimmArticle(
                    model_name=model_name.strip(),
                    paper_title=paper_title.strip(),
                    paper_url=paper_url.strip(),
                ),
            )
    return cleaned
