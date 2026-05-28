# Copyright (c) 2026 AIMER contributors.
"""Forgot password views."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, override

from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.models import User
from django.shortcuts import redirect
from django.utils import timezone

from auth.helpers import send_password_reset_email
from auth.models import Profile
from auth.security import audit_event
from auth.tokens import new_token_pair
from auth.views import AuthView

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse


class ForgetPasswordView(AuthView):
    """Request a password reset by email."""

    @override
    async def get(self, request: HttpRequest) -> HttpResponse:
        """
        Render form for non-authenticated users.

        Returns:
            HttpResponse: The rendered form or a redirect response.

        """
        if request.user.is_authenticated:
            return redirect("index")
        return await sync_to_async(super().get)(request)

    @override
    async def post(self, request: HttpRequest) -> HttpResponse:
        """
        Generate reset token and send reset email.

        Returns:
            HttpResponse: Redirect response to forgot-password page.

        """
        email = (request.POST.get("email") or "").strip().lower()
        if not email:
            await sync_to_async(audit_event)(
                "auth.password_reset.request_rejected",
                request=request,
                metadata={"reason": "missing_email"},
            )
            await sync_to_async(messages.error)(
                request,
                "Please enter your email address.",
            )
            return redirect("forgot-password")

        user = await User.objects.filter(email=email).afirst()
        if user:
            token, token_hash = new_token_pair()
            expiration_time = timezone.now() + timedelta(hours=24)
            user_profile, _created = await Profile.objects.aget_or_create(user=user)
            user_profile.forget_password_token = token_hash
            user_profile.forget_password_token_expires_at = expiration_time
            await user_profile.asave()
            await send_password_reset_email(email, token)
            await sync_to_async(audit_event)(
                "auth.password_reset.requested",
                request=request,
                user=user,
                actor_identifier=email,
            )
        else:
            await sync_to_async(audit_event)(
                "auth.password_reset.requested",
                request=request,
                actor_identifier=email,
                metadata={"account_found": False},
            )

        message_fn = messages.success
        message = "If an account exists for that email, a reset link has been sent."
        if not (settings.EMAIL_HOST_USER and settings.EMAIL_HOST_PASSWORD):
            message_fn = messages.error
            message = (
                "Email settings are not configured. Unable to send verification email."
            )
        await sync_to_async(message_fn)(request, message)
        return redirect("forgot-password")
