# Copyright (c) 2026 AIMER contributors.
"""Helper functions for authentication emails."""

from __future__ import annotations

import logging
from urllib.parse import urljoin

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.mail import EmailMessage
from django.urls import reverse

logger = logging.getLogger(__name__)


async def send_email(subject: str, email: str, message: str) -> None:
    """Send a plain-text email asynchronously."""
    email_from = getattr(settings, "DEFAULT_FROM_EMAIL", None) or getattr(
        settings,
        "EMAIL_HOST_USER",
        None,
    )
    if not email_from or not email:
        return

    email_message = EmailMessage(subject, message, email_from, [email])
    try:
        await sync_to_async(email_message.send)()
    except Exception:
        logger.exception("Failed to send email to %s.", email)


def get_absolute_url(path: str) -> str:
    """
    Build an absolute URL from a relative path.

    Returns:
        str: Absolute URL string.

    """
    return urljoin(settings.BASE_URL, path)


async def send_verification_email(email: str, token: str) -> None:
    """Send an email verification link."""
    verification_url = get_absolute_url(
        reverse("verify-email", kwargs={"token": token}),
    )
    message = f"Hi,\n\nPlease verify your email using this link: {verification_url}"
    await send_email("Verify your email", email, message)


async def send_password_reset_email(email: str, token: str) -> None:
    """Send a password reset link."""
    reset_url = get_absolute_url(reverse("reset-password", kwargs={"token": token}))
    message = f"Hi,\n\nPlease reset your password using this link: {reset_url}"
    await send_email("Reset your password", email, message)
