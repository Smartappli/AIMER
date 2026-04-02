#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""

import os
import sys

DJANGO_IMPORT_ERROR: ImportError | None = None

try:
    from django.core.management import execute_from_command_line
except ImportError as exc:
    DJANGO_IMPORT_ERROR = exc
    execute_from_command_line = None


def main() -> None:
    """Run administrative tasks.

    Raises:
        ImportError: If Django is not installed in the current environment.

    """
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "framework.settings")
    if execute_from_command_line is None:
        raise ImportError from DJANGO_IMPORT_ERROR
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
