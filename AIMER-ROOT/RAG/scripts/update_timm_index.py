# Copyright (c) 2026 AIMER contributors.
"""Utility script to refresh the TIMM article index from local PDF corpus."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def main() -> None:
    """
    Refresh the TIMM article index and report the resulting status.

    Returns:
        None.

    """
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    from RAG.timm_articles import (
        ensure_timm_article_index_is_fresh,
        load_timm_article_index,
    )

    refreshed = ensure_timm_article_index_is_fresh()
    rows = load_timm_article_index()
    status = "updated" if refreshed else "already up to date"
    sys.stdout.write(f"TIMM index {status}: {len(rows)} article(s).\n")


if __name__ == "__main__":
    main()
