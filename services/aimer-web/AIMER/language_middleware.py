# Copyright (c) 2026 AIMER contributors.
"""Middleware for default language cookie handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.conf import settings
from django.utils.translation import activate

if TYPE_CHECKING:
    from collections.abc import Callable

    from django.http import HttpRequest, HttpResponse


class DefaultLanguageMiddleware:
    """Set Django language cookie when it is missing."""

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        """Store next middleware/view callable."""
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        """
        Process request and ensure default language cookie exists.

        Returns:
            HTTP response with language cookie ensured.

        """
        if "django_language" not in request.COOKIES:
            default_language = settings.LANGUAGE_CODE
            activate(default_language)
            response = self.get_response(request)
            response.set_cookie("django_language", default_language)
            return response
        return self.get_response(request)
