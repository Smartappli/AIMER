# Copyright (c) 2026 AIMER contributors.
"""Tests for TIMM article index utilities."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, NoReturn

from RAG.timm_articles import (
    build_timm_article_index_from_pdfs,
    ensure_timm_article_index_is_fresh,
    load_timm_article_index,
)

if TYPE_CHECKING:
    from pathlib import Path


def _fail_test(message: str) -> NoReturn:
    """
    Raise a test failure with a consistent assertion message.

    Args:
        message: Failure message to report.

    Raises:
        AssertionError: Always raised with the provided message.

    """
    raise AssertionError(message)


def test_build_timm_article_index_from_pdfs_deduplicates_and_sorts(
    tmp_path: Path,
) -> None:
    """Building the index should deduplicate entries and keep them sorted."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    (
        pdf_dir / "vit_base_patch16_224 - An Image Is Worth 16x16 Words 2010.11929.pdf"
    ).write_text(
        "",
        encoding="utf-8",
    )
    (pdf_dir / "convnext_tiny - A ConvNet for the 2020s 2201.03545.pdf").write_text(
        "",
        encoding="utf-8",
    )
    (
        pdf_dir / "vit_base_patch16_224 - An Image Is Worth 16x16 Words 2010.11929.pdf"
    ).write_text(
        "",
        encoding="utf-8",
    )

    rows = build_timm_article_index_from_pdfs(pdf_directory=pdf_dir)

    expected = [
        {
            "model_name": "convnext_tiny",
            "paper_title": "convnext_tiny - A ConvNet for the 2020s",
            "paper_url": "https://arxiv.org/abs/2201.03545",
        },
        {
            "model_name": "vit_base_patch16_224",
            "paper_title": "vit_base_patch16_224 - An Image Is Worth 16x16 Words",
            "paper_url": "https://arxiv.org/abs/2010.11929",
        },
    ]

    if rows != expected:
        msg = f"Expected deduplicated sorted rows {expected!r}, got {rows!r}."
        _fail_test(msg)


def test_build_timm_article_index_from_pdfs_returns_empty_when_directory_missing(
    tmp_path: Path,
) -> None:
    """Building the index should return an empty list for a missing directory."""
    rows = build_timm_article_index_from_pdfs(pdf_directory=tmp_path / "missing")
    if rows != []:
        msg = f"Expected an empty list for missing directory, got {rows!r}."
        _fail_test(msg)


def test_load_timm_article_index_filters_invalid_rows(tmp_path: Path) -> None:
    """Loading the index should keep only valid TIMM article rows."""
    index_file = tmp_path / "timm_model_articles.json"
    index_file.write_text(
        json.dumps(
            [
                {
                    "model_name": "resnet50",
                    "paper_title": "Deep Residual Learning for Image Recognition",
                    "paper_url": "https://arxiv.org/abs/1512.03385",
                },
                {
                    "model_name": " ",
                    "paper_title": "Invalid",
                    "paper_url": "https://example.com",
                },
                "invalid",
            ],
        ),
        encoding="utf-8",
    )

    load_timm_article_index.cache_clear()
    rows = load_timm_article_index(index_file=index_file)

    expected = [
        {
            "model_name": "resnet50",
            "paper_title": "Deep Residual Learning for Image Recognition",
            "paper_url": "https://arxiv.org/abs/1512.03385",
        },
    ]

    if rows != expected:
        msg = f"Expected only valid rows {expected!r}, got {rows!r}."
        _fail_test(msg)


def test_ensure_timm_article_index_is_fresh_updates_stale_index(
    tmp_path: Path,
) -> None:
    """Refreshing should rewrite the index file when generated data changed."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (
        pdf_dir
        / "resnet50 - Deep Residual Learning for Image Recognition 1512.03385.pdf"
    ).write_text(
        "",
        encoding="utf-8",
    )

    index_file = tmp_path / "timm_model_articles.json"
    index_file.write_text("[]", encoding="utf-8")

    load_timm_article_index.cache_clear()
    refreshed = ensure_timm_article_index_is_fresh(
        pdf_directory=pdf_dir,
        index_file=index_file,
    )

    rows = json.loads(index_file.read_text(encoding="utf-8"))

    if refreshed is not True:
        msg = "Expected refreshed to be True."
        _fail_test(msg)

    row_count = len(rows)
    if row_count != 1:
        msg = f"Expected exactly one row, got {row_count}."
        _fail_test(msg)

    model_name = rows[0]["model_name"]
    if model_name != "resnet50":
        msg = f"Expected model_name 'resnet50', got {model_name!r}."
        _fail_test(msg)

    paper_url = rows[0]["paper_url"]
    if paper_url != "https://arxiv.org/abs/1512.03385":
        msg = (
            f"Expected paper_url 'https://arxiv.org/abs/1512.03385', got {paper_url!r}."
        )
        _fail_test(msg)
