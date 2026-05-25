# Copyright (c) 2026 AIMER contributors.
"""Django app configuration for auth app."""

from django.apps import AppConfig


class AuthConfig(AppConfig):
    """Configure the auth Django app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "auth"
    label = "accounts"
