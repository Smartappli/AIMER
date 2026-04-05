# Copyright (c) 2026 AIMER contributors.
"""Utility script to refresh the TIMM article index from local PDF corpus."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from RAG.timm_articles import ensure_timm_article_index_is_fresh, load_timm_article_index


def main() -> None:
    refreshed = ensure_timm_article_index_is_fresh()
    rows = load_timm_article_index()
    status = "updated" if refreshed else "already up to date"
    print(f"TIMM index {status}: {len(rows)} article(s).")


if __name__ == "__main__":
    main()
