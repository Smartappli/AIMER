# Copyright (c) 2026 AIMER contributors.
"""Authentication data models."""

from __future__ import annotations

from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver


class Profile(models.Model):
    """Additional profile information attached to a Django user."""

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="profile",
    )
    email = models.EmailField(max_length=100, unique=True)
    email_token = models.CharField(max_length=100, blank=True, null=True)
    email_token_expires_at = models.DateTimeField(blank=True, null=True)
    forget_password_token = models.CharField(max_length=100, blank=True, null=True)
    forget_password_token_expires_at = models.DateTimeField(blank=True, null=True)
    is_verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        """Django model metadata."""

        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"

    def __str__(self) -> str:
        """
        Return a readable username for admin pages.

        Returns:
            str: The related user's username.

        """
        return self.user.username


class SecurityAuditEvent(models.Model):
    """Append-only security event for authentication and sensitive actions."""

    event_type = models.CharField(max_length=80, db_index=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
        related_name="security_audit_events",
    )
    actor_identifier = models.CharField(max_length=255, blank=True)
    ip_address = models.GenericIPAddressField(blank=True, null=True)
    user_agent = models.TextField(blank=True)
    path = models.CharField(max_length=512, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        """Django model metadata."""

        ordering = ["-created_at"]
        verbose_name = "Security audit event"
        verbose_name_plural = "Security audit events"

    def __str__(self) -> str:
        """
        Return a concise event label for admin pages.

        Returns:
            str: Event type and actor identifier.

        """
        actor = self.actor_identifier or (self.user_id and str(self.user_id)) or "-"
        return f"{self.event_type}:{actor}"


@receiver(post_save, sender=User)
def create_profile(
    sender: type[User],
    instance: User,
    created: object,
    **_kwargs: object,
) -> None:
    """Create a profile automatically after user creation."""
    del sender
    if bool(created):
        Profile.objects.create(user=instance, email=instance.email)
