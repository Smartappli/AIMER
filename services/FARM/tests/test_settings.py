# Copyright (c) 2026 AIMER contributors.
"""Tests for FARM production configuration validation."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("DJANGO_SECRET_KEY", "test-secret-key")
os.environ.setdefault("DJANGO_ENVIRONMENT", "local")
os.environ.setdefault("DJANGO_SECURE_SSL_REDIRECT", "0")

from FARM import settings as farm_settings


STRONG_SECRET = "prod-secret-abcdefghijklmnopqrstuvwxyz-0123456789-ABCDEFGHIJKLMN"


def _set_safe_production_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch FARM settings globals to a valid production baseline."""
    monkeypatch.setattr(farm_settings, "IS_PRODUCTION", True)
    monkeypatch.setattr(farm_settings, "DEBUG", False)
    monkeypatch.setattr(farm_settings, "SECRET_KEY", STRONG_SECRET)
    monkeypatch.setattr(farm_settings, "ALLOWED_HOSTS", ["farm.example.org"])
    monkeypatch.setattr(farm_settings, "BASE_URL", "https://farm.example.org")
    monkeypatch.setattr(
        farm_settings,
        "DATABASE_URL",
        "postgresql://farm:password@db.example.org:5432/farm?sslmode=require",
    )
    monkeypatch.setattr(farm_settings, "SECURE_SSL_REDIRECT", True)
    monkeypatch.setattr(farm_settings, "SESSION_COOKIE_SECURE", True)
    monkeypatch.setattr(farm_settings, "CSRF_COOKIE_SECURE", True)
    monkeypatch.setattr(farm_settings, "SECURE_HSTS_PRELOAD", True)
    monkeypatch.setattr(farm_settings, "SECURE_HSTS_SECONDS", 31536000)


def test_production_configuration_accepts_safe_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure a hardened FARM production baseline passes validation."""
    _set_safe_production_defaults(monkeypatch)

    farm_settings.validate_production_configuration()


def test_production_configuration_requires_explicit_allowed_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure production cannot rely on implicit localhost hosts."""
    _set_safe_production_defaults(monkeypatch)
    monkeypatch.setattr(farm_settings, "ALLOWED_HOSTS", [])

    with pytest.raises(RuntimeError, match="DJANGO_ALLOWED_HOSTS"):
        farm_settings.validate_production_configuration()


def test_production_configuration_requires_public_https_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure production FARM URLs are explicit public HTTPS origins."""
    _set_safe_production_defaults(monkeypatch)
    monkeypatch.setattr(farm_settings, "BASE_URL", "http://localhost:8000")

    with pytest.raises(RuntimeError, match="public HTTPS URL"):
        farm_settings.validate_production_configuration()


def test_production_configuration_rejects_weak_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure short or low-entropy secrets cannot pass production validation."""
    _set_safe_production_defaults(monkeypatch)
    monkeypatch.setattr(farm_settings, "SECRET_KEY", "short")

    with pytest.raises(RuntimeError, match="high entropy"):
        farm_settings.validate_production_configuration()


def test_production_configuration_rejects_sqlite_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure regulated production cannot run on local SQLite storage."""
    _set_safe_production_defaults(monkeypatch)
    monkeypatch.setattr(
        farm_settings,
        "DATABASE_URL",
        "sqlite:///tmp/farm.sqlite3",
    )

    with pytest.raises(RuntimeError, match="PostgreSQL"):
        farm_settings.validate_production_configuration()
