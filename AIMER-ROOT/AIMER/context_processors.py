# Copyright (c) 2026 AIMER contributors.
"""Context processors exposed to Django templates."""

from __future__ import annotations

from typing import Any

from django.conf import settings
from django.http import HttpRequest


def my_setting(_request: HttpRequest) -> dict[str, Any]:
    """Add Django settings to template context."""
    return {"MY_SETTING": settings}


def language_code(request: HttpRequest) -> dict[str, str]:
    """Add active request language code to template context."""
    return {"LANGUAGE_CODE": request.LANGUAGE_CODE}


def get_cookie(request: HttpRequest) -> dict[str, Any]:
    """Expose request cookies in template context."""
    return {"COOKIES": request.COOKIES}


def environment(_request: HttpRequest) -> dict[str, str]:
    """Expose runtime environment in template context."""
    return {"ENVIRONMENT": settings.ENVIRONMENT}
