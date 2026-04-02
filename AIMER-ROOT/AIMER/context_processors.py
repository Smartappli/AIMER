# Copyright (c) 2026 AIMER contributors.
"""Context processors exposed to Django templates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.conf import settings

if TYPE_CHECKING:
    from django.http import HttpRequest


def my_setting(_request: HttpRequest) -> dict[str, Any]:
    """Add Django settings to template context.

    Returns:
        Mapping with Django settings object.

    """
    return {"MY_SETTING": settings}


def language_code(request: HttpRequest) -> dict[str, str]:
    """Add active request language code to template context.

    Returns:
        Mapping with active language code.

    """
    return {"LANGUAGE_CODE": request.LANGUAGE_CODE}


def get_cookie(request: HttpRequest) -> dict[str, Any]:
    """Expose request cookies in template context.

    Returns:
        Mapping with cookie dictionary.

    """
    return {"COOKIES": request.COOKIES}


def environment(_request: HttpRequest) -> dict[str, str]:
    """Expose runtime environment in template context.

    Returns:
        Mapping with the configured environment name.

    """
    return {"ENVIRONMENT": settings.ENVIRONMENT}
