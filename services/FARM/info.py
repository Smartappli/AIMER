# Copyright (C) 2026 AIMER contributors.

"""Display installed package versions used by FARM integrations."""

from importlib.metadata import PackageNotFoundError, version
import sys


def _safe_version(package_name: str) -> str:
    """Return package version or a deterministic missing marker."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"


def main() -> None:
    """Print Flower package version used by FARM."""
    sys.stdout.write(f"flwr.__version__={_safe_version('flwr')}\n")


if __name__ == "__main__":
    main()
