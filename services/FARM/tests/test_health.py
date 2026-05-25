# Copyright (C) 2026 AIMER contributors.

"""Tests for FARM deployment health checks."""

from __future__ import annotations

import os
from http import HTTPStatus

import django
from django.test import Client

os.environ.setdefault("DJANGO_SECRET_KEY", "test-secret-key")
os.environ.setdefault("DJANGO_SECURE_SSL_REDIRECT", "0")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "FARM.settings")
django.setup()


def test_healthz_returns_service_status() -> None:
    """Ensure smoke checks can verify the FARM service without auth."""
    response = Client(HTTP_HOST="localhost").get("/healthz/")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"service": "FARM", "status": "ok"}


def test_healthz_rejects_unsafe_methods() -> None:
    """Ensure the deployment smoke endpoint is read-only."""
    response = Client(HTTP_HOST="localhost").post("/healthz/")

    assert response.status_code == HTTPStatus.METHOD_NOT_ALLOWED
