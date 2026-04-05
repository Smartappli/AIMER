# Copyright (C) 2026 AIMER contributors.

"""Shared helpers and constants for API microservices."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Final

LIBRARIES_BY_CATEGORY: Final[dict[str, dict[str, str]]] = {
    "AI": {
        "keras": "keras",
        "segmentation-models-pytorch": "segmentation_models_pytorch",
        "tensorflow": "tensorflow",
        "timm": "timm",
        "torch": "torch",
    },
}


def safe_version(pkg_name: str, module_name: str | None = None) -> str | None:
    """
    Safely resolve a library version.

    Args:
        pkg_name: The distribution/package name as known by packaging metadata.
        module_name: Optional importable module name to check ``__version__``.

    Returns:
        The detected version string, or ``None`` if the package/module is absent
        or any error occurs.

    """
    try:
        if module_name:
            module = __import__(module_name)
            detected = getattr(module, "__version__", None)
            if isinstance(detected, str) and detected:
                return detected
        return version(pkg_name)
    except PackageNotFoundError:
        return None
    except (ImportError, AttributeError):
        return None
