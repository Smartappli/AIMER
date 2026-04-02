# Copyright (c) 2026 AIMER contributors.
"""Forgot password views."""

from __future__ import annotations

import uuid
from datetime import timedelta

from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.models import User
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect
from django.utils import timezone

from auth.helpers import send_password_reset_email
from auth.models import Profile
from auth.views import AuthView


class ForgetPasswordView(AuthView):
    """Request a password reset by email."""

    async def get(self, request: HttpRequest) -> HttpResponse:
        """Render form for non-authenticated users."""
        if request.user.is_authenticated:
            return redirect("index")
        return await sync_to_async(super().get)(request)

    async def post(self, request: HttpRequest) -> HttpResponse:
        """Generate reset token and send reset email."""
        email = request.POST.get("email")
        if not email:
            await sync_to_async(messages.error)(
                request,
                "Please enter your email address.",
            )
            return redirect("forgot-password")

        user = await User.objects.filter(email=email).afirst()
        if user:
            token = str(uuid.uuid4())
            expiration_time = timezone.now() + timedelta(hours=24)
            user_profile, _created = await Profile.objects.aget_or_create(user=user)
            user_profile.forget_password_token = token
            user_profile.forget_password_token_expires_at = expiration_time
            await user_profile.asave()
            await send_password_reset_email(email, token)

        message_fn = messages.success
        message = "If an account exists for that email, a reset link has been sent."
        if not (settings.EMAIL_HOST_USER and settings.EMAIL_HOST_PASSWORD):
            message_fn = messages.error
            message = (
                "Email settings are not configured. Unable to send verification email."
            )
        await sync_to_async(message_fn)(request, message)
        return redirect("forgot-password")
