# Copyright (c) 2026 AIMER contributors.
"""Utilities to expose the default TIMM paper index for the RAG knowledge base."""

from __future__ import annotations

import json
import re
from functools import lru_cache
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


def timm_pdf_directory() -> Path:
    """Return the TIMM PDF corpus location used to build/update the index."""
    return Path(__file__).resolve().parent / "data" / "pdfs"


def _extract_article_from_pdf_name(filename: str) -> TimmArticle:
    stem = Path(filename).stem
    model_name = stem.split(" - ")[0].strip()

    arxiv_match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", stem)
    if arxiv_match:
        paper_url = f"https://arxiv.org/abs/{arxiv_match.group(1)}"
        paper_title = stem[: arxiv_match.start()].rstrip(" -_")
    else:
        paper_url = "https://github.com/huggingface/pytorch-image-models"
        paper_title = model_name

    return TimmArticle(
        model_name=model_name,
        paper_title=paper_title,
        paper_url=paper_url,
    )


def build_timm_article_index_from_pdfs(pdf_directory: Path | None = None) -> list[TimmArticle]:
    """
    Build a normalized TIMM index from local PDF filenames.

    Returns:
        Deduplicated list sorted by model name then paper title.

    """
    base_dir = pdf_directory or timm_pdf_directory()
    if not base_dir.exists():
        return []

    rows = [_extract_article_from_pdf_name(path.name) for path in sorted(base_dir.glob("*.pdf"))]
    dedup: dict[tuple[str, str], TimmArticle] = {}
    for row in rows:
        key = (row["model_name"], row["paper_url"])
        dedup[key] = row

    return sorted(
        dedup.values(),
        key=lambda article: (article["model_name"].lower(), article["paper_title"].lower()),
    )


def save_timm_article_index(
    rows: list[TimmArticle],
    *,
    index_file: Path | None = None,
) -> None:
    """Save TIMM rows in a deterministic JSON format."""
    path = index_file or timm_article_index_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ensure_timm_article_index_is_fresh(
    *,
    pdf_directory: Path | None = None,
    index_file: Path | None = None,
) -> bool:
    """
    Automatically refresh the TIMM index from PDF corpus when it is stale.

    Returns:
        ``True`` when a refresh happened, otherwise ``False``.

    """
    target_file = index_file or timm_article_index_file()
    generated = build_timm_article_index_from_pdfs(pdf_directory=pdf_directory)
    if not generated:
        return False

    current = load_timm_article_index(index_file=target_file)
    if current == generated:
        return False

    save_timm_article_index(generated, index_file=target_file)
    load_timm_article_index.cache_clear()
    return True


@lru_cache(maxsize=4)
def load_timm_article_index(index_file: Path | None = None) -> list[TimmArticle]:
    """
    Load the default TIMM paper references bundled with the repository.

    Returns:
        List of TIMM paper references. Empty list when the file is missing or invalid.

    """
    path = index_file or timm_article_index_file()
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
