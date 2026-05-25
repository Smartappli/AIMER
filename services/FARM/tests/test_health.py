# Copyright (C) 2026 AIMER contributors.

"""Tests for FARM deployment health checks."""

from __future__ import annotations

import os

import django
from django.test import Client

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "FARM.settings")
django.setup()


def test_healthz_returns_service_status() -> None:
    """Ensure smoke checks can verify the FARM service without auth."""
    response = Client(HTTP_HOST="localhost").get("/healthz/")

    assert response.status_code == 200
    assert response.json() == {"service": "FARM", "status": "ok"}
