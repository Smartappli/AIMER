# Copyright (c) 2026 AIMER contributors.
"""Unit tests for TIMM index automation and OMOP metadata helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from RAG.omop import build_omop_metadata
from RAG.timm_articles import (
    build_timm_article_index_from_pdfs,
    ensure_timm_article_index_is_fresh,
    load_timm_article_index,
)


def test_build_timm_article_index_from_pdfs_is_deterministic(tmp_path: Path) -> None:
    """Index generation should be sorted and parse arXiv ids from filenames."""
    (tmp_path / "ViT - 2010.11929v2.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "ResNet - 1512.03385v1.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "ResNet - 1512.03385v1.pdf").write_text("x", encoding="utf-8")

    rows = build_timm_article_index_from_pdfs(tmp_path)

    assert rows[0]["model_name"] == "ResNet"
    assert rows[0]["paper_url"].endswith("1512.03385v1")
    assert rows[1]["model_name"] == "ViT"


def test_ensure_timm_article_index_is_fresh_updates_stale_json(tmp_path: Path) -> None:
    """A stale JSON index should be replaced by data generated from PDFs."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "Swin Transformer - 2103.14030v2.pdf").write_text("x", encoding="utf-8")

    index_file = tmp_path / "timm_model_articles.json"
    index_file.write_text("[]", encoding="utf-8")

    refreshed = ensure_timm_article_index_is_fresh(
        pdf_directory=pdf_dir,
        index_file=index_file,
    )
    rows = load_timm_article_index(index_file=index_file)

    assert refreshed is True
    assert len(rows) == 1
    assert rows[0]["model_name"] == "Swin Transformer"
    assert json.loads(index_file.read_text(encoding="utf-8"))[0]["paper_url"].endswith(
        "2103.14030v2",
    )


def test_omop_metadata_contains_expected_fields() -> None:
    """OMOP helper should extract concept ids and SNOMED codes from clinical text."""
    metadata = build_omop_metadata("classification MRI for pneumonia with AUC")

    assert "omop_condition_concept_ids" in metadata
    assert "omop_modality_concept_ids" in metadata
    assert "omop_measurement_concept_ids" in metadata
    assert "snomed_ct_codes" in metadata
