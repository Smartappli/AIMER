# Copyright (C) 2026 AIMER contributors.

"""Tests for FARM deployment health checks."""

from __future__ import annotations

import os
from http import HTTPStatus
from unittest.mock import patch

import django
import pytest
from django.test import Client

import manage

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


def test_manage_uses_farm_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure container entrypoint management commands load FARM settings."""
    monkeypatch.delenv("DJANGO_SETTINGS_MODULE", raising=False)

    with patch("manage.execute_from_command_line") as mock_execute:
        manage.main()

    assert os.environ["DJANGO_SETTINGS_MODULE"] == "FARM.settings"
    mock_execute.assert_called_once()
