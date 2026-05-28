# Copyright (c) 2026 AIMER contributors.
"""Admin registrations for auth app."""

from django.contrib import admin

from auth.models import Profile, SecurityAuditEvent


@admin.register(Profile)
class Member(admin.ModelAdmin):
    """Admin configuration for Profile."""

    list_display = ("user", "email", "is_verified", "created_at")


@admin.register(SecurityAuditEvent)
class SecurityAuditEventAdmin(admin.ModelAdmin):
    """Read-only admin view for security audit events."""

    list_display = ("created_at", "event_type", "user", "actor_identifier", "path")
    list_filter = ("event_type", "created_at")
    search_fields = ("actor_identifier", "user__username", "user__email", "path")
    readonly_fields = (
        "created_at",
        "event_type",
        "user",
        "actor_identifier",
        "ip_address",
        "user_agent",
        "path",
        "metadata",
    )

    def has_add_permission(self, _request: object) -> bool:
        """Disable manual creation from admin."""
        return False

    def has_change_permission(
        self,
        _request: object,
        _obj: object | None = None,
    ) -> bool:
        """Disable mutation from admin."""
        return False

    def has_delete_permission(
        self,
        _request: object,
        _obj: object | None = None,
    ) -> bool:
        """Disable deletion from admin."""
        return False
