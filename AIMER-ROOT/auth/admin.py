# Copyright (c) 2026 AIMER contributors.
"""Admin registrations for auth app."""

from django.contrib import admin

from auth.models import Profile


@admin.register(Profile)
class Member(admin.ModelAdmin):
    """Admin configuration for Profile."""

    list_display = ("user", "email", "is_verified", "created_at")
