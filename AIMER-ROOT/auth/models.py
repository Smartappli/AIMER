# Copyright (c) 2026 AIMER contributors.
"""Authentication data models."""

from __future__ import annotations

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
    email_token = models.CharField(max_length=100, blank=True, default="")
    forget_password_token = models.CharField(max_length=100, blank=True, default="")
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
