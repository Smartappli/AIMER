# Copyright (c) 2026 AIMER contributors.
"""Middleware for security-sensitive audit events."""

from __future__ import annotations

from collections.abc import Callable

from django.http import HttpRequest, HttpResponse

from auth.security import audit_event

ADMIN_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class AdminAuditMiddleware:
    """Audit privileged admin mutations for SIEM and compliance evidence."""

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        """Store the next middleware/view callable."""
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        """Audit authenticated staff mutations under the Django admin path."""
        response = self.get_response(request)
        if self._should_audit(request):
            audit_event(
                "admin.privileged_action",
                request=request,
                user=request.user,
                metadata={
                    "method": request.method,
                    "status_code": response.status_code,
                },
            )
        return response

    @staticmethod
    def _should_audit(request: HttpRequest) -> bool:
        """Return whether the request is a privileged admin mutation."""
        user = getattr(request, "user", None)
        return bool(
            request.path.startswith("/admin/")
            and request.method in ADMIN_MUTATING_METHODS
            and getattr(user, "is_authenticated", False)
            and getattr(user, "is_staff", False)
        )
