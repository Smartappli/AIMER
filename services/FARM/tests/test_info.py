# Copyright (C) 2026 AIMER contributors.

"""Tests for FARM package version reporting helpers."""

from importlib.metadata import PackageNotFoundError
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from info import _safe_version


def test_safe_version_returns_marker_when_missing(monkeypatch) -> None:
    """Ensure missing package metadata yields deterministic marker."""

    def fake_version(_package_name: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("info.version", fake_version)
    assert _safe_version("flwr") == "NOT_INSTALLED"
