import logging
from urllib.parse import urljoin

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.mail import EmailMessage
from django.urls import reverse

logger = logging.getLogger(__name__)


async def send_email(subject, email, message):
    """Send an email asynchronously using Django's synchronous email backend.

    This helper wraps `EmailMessage.send()` (a blocking call) with
    `asgiref.sync.sync_to_async()` so it can be awaited from async views/tasks.

    Args:
        subject: Email subject line.
        email: Recipient email address.
        message: Plain-text email body.

    Notes:
        - Errors are logged with the module logger to aid diagnostics.

    """
    email_from = getattr(settings, "DEFAULT_FROM_EMAIL", None) or getattr(
        settings,
        "EMAIL_HOST_USER",
        None,
    )
    if not email_from:
        logger.warning("Email sender is not configured; skipping send.")
        return
    if not email:
        logger.warning("No recipient email provided; skipping send.")
        return

    recipient_list = [email]
    email_message = EmailMessage(subject, message, email_from, recipient_list)
    try:
        await sync_to_async(email_message.send)()
    except Exception:
        logger.exception("Failed to send email to %s.", email)


def get_absolute_url(path):
    """Build an absolute URL from a relative path.

    Args:
        path: A relative URL path (e.g. "/verify/abc123/") typically returned by
            `django.urls.reverse()`.

    Returns:
        An absolute URL combining `settings.BASE_URL` and the provided path.

    """
    return urljoin(settings.BASE_URL, path)


async def send_verification_email(email, token):
    """Send an email verification link to a user.

    Args:
        email: Recipient email address.
        token: Verification token embedded into the URL.

    Side effects:
        Sends an email with a link pointing to the `verify-email` route.

    """
    subject = "Verify your email"
    verification_url = get_absolute_url(
        reverse("verify-email", kwargs={"token": token}),
    )
    message = (
        f"Hi,\n\nPlease verify your email using this link: {verification_url}"
    )
    await send_email(subject, email, message)


async def send_password_reset_email(email, token):
    """Send a password reset link to a user.

    Args:
        email: Recipient email address.
        token: Password reset token embedded into the URL.

    Side effects:
        Sends an email with a link pointing to the `reset-password` route.

    """
    subject = "Reset your password"
    reset_url = get_absolute_url(
        reverse("reset-password", kwargs={"token": token}),
    )
    message = f"Hi,\n\nPlease reset your password using this link: {reset_url}"
    await send_email(subject, email, message)
