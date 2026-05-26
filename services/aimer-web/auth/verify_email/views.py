# Copyright (c) 2026 AIMER contributors.
"""Email verification views."""

import secrets
from typing import override

from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib import messages
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from django.shortcuts import redirect

from auth.helpers import send_verification_email
from auth.models import Profile
from auth.views import AuthView


class VerifyEmailTokenView(AuthView):
    """
    Verify a user's email address using a token.

    GET /verify-email/<token>:
    - Finds the Profile with `email_token == token`.
    - Sets `is_verified = True` and clears `email_token`.
    - Shows a success message (mainly for non-authenticated users).
    - Redirects to the login page.

    Error handling:
    - If the token is invalid, shows an error message and redirects to the
      verify email page.
    """

    @override
    async def get(self, request: HttpRequest, token: str) -> HttpResponseRedirect:
        """
        Handle token verification.

        Args:
            request: Django HttpRequest.
            token: Verification token extracted from the URL.

        Returns:
            An HTTP redirect response.

        """
        profile = await Profile.objects.filter(email_token=token).afirst()
        if not profile:
            await sync_to_async(messages.error)(
                request,
                "Invalid token, please try again",
            )
            return redirect("verify-email-page")

        profile.is_verified = True
        profile.email_token = None
        await profile.asave()
        if not request.user.is_authenticated:
            # User is not already authenticated
            # Perform the email verification and any other necessary actions
            await sync_to_async(messages.success)(
                request,
                "Email verified successfully",
            )
        return redirect("login")
        # Now, redirect to the login page


class VerifyEmailView(AuthView):
    """
    Display the verify email page.

    This is typically a page where users are informed that they must verify their
    email address and where they can trigger a resend of the verification email.
    """

    @override
    async def get(self, request: HttpRequest) -> HttpResponse:
        """
        Render the verify email page.

        Args:
            request: Django HttpRequest.

        Returns:
            The response returned by the parent AuthView's GET handler.

        """
        return await sync_to_async(super().get)(request)


class SendVerificationView(AuthView):
    """
    Generate and send a verification email.

    POST /send-verification:
    - Determines the target email (authenticated user profile or session).
    - Generates a new token, saves it into Profile.email_token.
    - Sends the verification email.
    - Displays a success or error message.
    - Redirects back to the verify email page.
    """

    http_method_names = ["post"]

    @override
    async def post(self, request: HttpRequest) -> HttpResponseRedirect:
        """
        Send a (re)verification email to the user.

        Args:
            request: Django HttpRequest.

        Returns:
            An HTTP redirect response to the verify email page.

        """
        email = await self.get_email(request)

        if not email:
            await sync_to_async(messages.error)(
                request,
                "Email not found in session",
            )
            return redirect("verify-email-page")

        if not (settings.EMAIL_HOST_USER and settings.EMAIL_HOST_PASSWORD):
            await sync_to_async(messages.error)(
                request,
                "Email settings are not configured. Unable to send verification email.",
            )
            return redirect("verify-email-page")

        user_profile = await Profile.objects.filter(email=email).afirst()
        if not user_profile:
            await sync_to_async(messages.error)(
                request,
                "Unable to find a profile for that email.",
            )
            return redirect("verify-email-page")

        if user_profile.is_verified:
            await sync_to_async(messages.success)(request, "Email is already verified.")
            return redirect("verify-email-page")

        token = secrets.token_urlsafe(32)
        user_profile.email_token = token
        await user_profile.asave()
        await send_verification_email(email, token)
        await sync_to_async(messages.success)(
            request,
            "Verification email sent successfully",
        )

        return redirect("verify-email-page")

    @staticmethod
    async def get_email(
        request: HttpRequest,
    ) -> str | None:
        """
        Resolve the recipient email.

        Rules:
        - If authenticated, use `request.user.profile.email`.
        - If not authenticated, use `request.session["email"]` if present.
        Returns:
            The target email address, or ``None`` when it cannot be resolved.

        """
        if request.user.is_authenticated:
            profile = await Profile.objects.filter(user=request.user).afirst()
            return profile.email if profile else None

        email = request.session.get("email")
        return str(email) if email else None
