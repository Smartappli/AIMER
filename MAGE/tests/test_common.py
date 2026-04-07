# Copyright (C) 2026 AIMER contributors.

"""Unit tests for shared API helpers in ``MAGE/api/common.py``."""

from __future__ import annotations

import importlib.metadata as md
import sys
from types import ModuleType
from typing import TYPE_CHECKING

from MAGE.api import common

if TYPE_CHECKING:
    import pytest


class DummyModuleWithVersion(ModuleType):
    """Dummy module exposing a usable ``__version__`` string."""

    def __init__(self) -> None:
        """Create deterministic dummy metadata."""
        super().__init__("dummy_mod")
        self.__version__ = "1.2.3"


class DummyModuleWithoutVersion(ModuleType):
    """Dummy module without any ``__version__`` attribute."""

    def __init__(self) -> None:
        """Create module without explicit version metadata."""
        super().__init__("dummy_no_ver")


def check(condition: object, message: str) -> None:
    """
    Raise an error if a condition is false.

    Raises:
        AssertionError: If ``condition`` is falsy.

    """
    if not condition:
        raise AssertionError(message)


def test_safe_version_prefers_module_dunder_version() -> None:
    """When module has ``__version__``, it should be returned directly."""
    dummy = DummyModuleWithVersion()
    sys.modules[dummy.__name__] = dummy

    detected = common.safe_version("package-does-not-matter", dummy.__name__)

    check(detected == "1.2.3", "Expected module __version__ to be returned")


def test_safe_version_falls_back_to_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no module name is provided, metadata version should be used."""

    def fake_version(pkg_name: str) -> str:
        check(pkg_name == "some-package", "Unexpected package name")
        return "9.9.9"

    monkeypatch.setattr(common, "version", fake_version)

    detected = common.safe_version("some-package")

    check(detected == "9.9.9", "Expected distribution metadata version")


def test_safe_version_falls_back_when_module_has_no_dunder_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If module exists but lacks ``__version__``, metadata lookup is used."""
    dummy = DummyModuleWithoutVersion()
    sys.modules[dummy.__name__] = dummy

    monkeypatch.setattr(common, "version", lambda _: "2.0.0")

    detected = common.safe_version("dummy-package", dummy.__name__)

    check(detected == "2.0.0", "Expected metadata fallback version")


def test_safe_version_returns_none_when_distribution_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing distributions should resolve to ``None``."""

    def fake_version(_: str) -> str:
        raise md.PackageNotFoundError

    monkeypatch.setattr(common, "version", fake_version)

    detected = common.safe_version("missing-package")

    check(detected is None, "Expected None for missing distribution")


def test_safe_version_returns_none_on_module_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import errors from optional modules should return ``None``."""
    real_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "broken_module":
            msg = "module unavailable"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    detected = common.safe_version("ignored-package", "broken_module")

    check(detected is None, "Expected None when module import fails")
