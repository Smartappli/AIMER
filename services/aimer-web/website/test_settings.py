# Copyright (c) 2026 AIMER contributors.
"""Tests for AIMER production configuration validation."""

from __future__ import annotations

from AIMER import settings as aimer_settings
from django.test import SimpleTestCase


STRONG_SECRET = "prod-secret-abcdefghijklmnopqrstuvwxyz-0123456789-ABCDEFGHIJKLMN"


class ProductionSettingsTests(SimpleTestCase):
    """Production settings must reject non-resilient deployment modes."""

    def _set_safe_production_defaults(self) -> None:
        """Patch AIMER settings globals to a valid production baseline."""
        self.addCleanup(
            setattr, aimer_settings, "IS_PRODUCTION", aimer_settings.IS_PRODUCTION
        )
        self.addCleanup(setattr, aimer_settings, "DEBUG", aimer_settings.DEBUG)
        self.addCleanup(
            setattr, aimer_settings, "SECRET_KEY", aimer_settings.SECRET_KEY
        )
        self.addCleanup(
            setattr, aimer_settings, "ALLOWED_HOSTS", aimer_settings.ALLOWED_HOSTS
        )
        self.addCleanup(setattr, aimer_settings, "BASE_URL", aimer_settings.BASE_URL)
        self.addCleanup(
            setattr, aimer_settings, "DATABASE_URL", aimer_settings.DATABASE_URL
        )
        self.addCleanup(
            setattr,
            aimer_settings,
            "SECURE_SSL_REDIRECT",
            aimer_settings.SECURE_SSL_REDIRECT,
        )
        self.addCleanup(
            setattr,
            aimer_settings,
            "SESSION_COOKIE_SECURE",
            aimer_settings.SESSION_COOKIE_SECURE,
        )
        self.addCleanup(
            setattr,
            aimer_settings,
            "CSRF_COOKIE_SECURE",
            aimer_settings.CSRF_COOKIE_SECURE,
        )
        self.addCleanup(
            setattr,
            aimer_settings,
            "SECURE_HSTS_PRELOAD",
            aimer_settings.SECURE_HSTS_PRELOAD,
        )
        self.addCleanup(
            setattr,
            aimer_settings,
            "SECURE_HSTS_SECONDS",
            aimer_settings.SECURE_HSTS_SECONDS,
        )
        self.addCleanup(
            setattr,
            aimer_settings,
            "EMAIL_VERIFICATION_REQUIRED",
            aimer_settings.EMAIL_VERIFICATION_REQUIRED,
        )
        self.addCleanup(
            setattr,
            aimer_settings,
            "EMAIL_HOST_USER",
            aimer_settings.EMAIL_HOST_USER,
        )
        self.addCleanup(
            setattr,
            aimer_settings,
            "EMAIL_HOST_PASSWORD",
            aimer_settings.EMAIL_HOST_PASSWORD,
        )
        self.addCleanup(
            setattr,
            aimer_settings,
            "RAG_SERVICE_URL",
            aimer_settings.RAG_SERVICE_URL,
        )
        self.addCleanup(
            setattr,
            aimer_settings,
            "RAG_SERVICE_CA_CERT_PATH",
            aimer_settings.RAG_SERVICE_CA_CERT_PATH,
        )

        aimer_settings.IS_PRODUCTION = True
        aimer_settings.DEBUG = False
        aimer_settings.SECRET_KEY = STRONG_SECRET
        aimer_settings.ALLOWED_HOSTS = ["aimer.example.org"]
        aimer_settings.BASE_URL = "https://aimer.example.org"
        aimer_settings.DATABASE_URL = (
            "postgresql://aimer:password@db.example.org:5432/aimer?sslmode=require"
        )
        aimer_settings.SECURE_SSL_REDIRECT = True
        aimer_settings.SESSION_COOKIE_SECURE = True
        aimer_settings.CSRF_COOKIE_SECURE = True
        aimer_settings.SECURE_HSTS_PRELOAD = True
        aimer_settings.SECURE_HSTS_SECONDS = 31536000
        aimer_settings.EMAIL_VERIFICATION_REQUIRED = True
        aimer_settings.EMAIL_HOST_USER = "smtp-user"
        aimer_settings.EMAIL_HOST_PASSWORD = "smtp-password"
        aimer_settings.RAG_SERVICE_URL = ""
        aimer_settings.RAG_SERVICE_CA_CERT_PATH = ""

    def test_production_configuration_accepts_safe_baseline(self) -> None:
        """A hardened production baseline passes validation."""
        self._set_safe_production_defaults()

        aimer_settings.validate_production_configuration()

    def test_production_configuration_rejects_sqlite_database_url(self) -> None:
        """Production must use resilient PostgreSQL storage, not SQLite."""
        self._set_safe_production_defaults()
        aimer_settings.DATABASE_URL = "sqlite:///tmp/aimer.sqlite3"

        with self.assertRaisesRegex(RuntimeError, "PostgreSQL"):
            aimer_settings.validate_production_configuration()

    def test_production_configuration_rejects_http_rag_service(self) -> None:
        """Production RAG requests must not traverse an unencrypted link."""
        self._set_safe_production_defaults()
        aimer_settings.RAG_SERVICE_URL = "http://aimer-rag.internal:8000"
        aimer_settings.RAG_SERVICE_CA_CERT_PATH = "/etc/aimer-rag-ca/ca.crt"

        with self.assertRaisesRegex(RuntimeError, "RAG_SERVICE_URL must use HTTPS"):
            aimer_settings.validate_production_configuration()
