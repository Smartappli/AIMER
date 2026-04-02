# Copyright (C) 2026 AIMER contributors.

"""Display installed package versions used by FARM integrations."""

import sys

import syft_client as sc
import syft_flwr


def main() -> None:
    """Print Syft-related package versions."""
    sys.stdout.write(f"sc.__version__={sc.__version__}\n")
    sys.stdout.write(f"syft_flwr.__version__={syft_flwr.__version__}\n")


if __name__ == "__main__":
    main()
