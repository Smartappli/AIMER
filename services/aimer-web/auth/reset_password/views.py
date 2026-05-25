# Copyright (c) 2026 AIMER contributors.
"""Password reset views."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from asgiref.sync import sync_to_async
from django.contrib import messages
from django.contrib.auth import aauthenticate, alogin
from django.shortcuts import redirect, render
from django.utils import timezone

from auth.models import Profile
from auth.views import AuthView

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse


class ResetPasswordView(AuthView):
    """Handle password reset form rendering and submission."""

    @override
    async def get(self, request: HttpRequest, _token: str) -> HttpResponse:
        """
        Render reset-password form for anonymous users.

        Returns:
            HttpResponse: Reset-password page or redirect for authenticated users.

        """
        if request.user.is_authenticated:
            return redirect("index")

        return await sync_to_async(super().get)(request)

    @override
    async def post(self, request: HttpRequest, token: str) -> HttpResponse:
        """
        Validate token, set a new password, then login user if possible.

        Returns:
            HttpResponse: Redirect or rendered template based on validation result.

        """
        profile = await Profile.objects.filter(
            forget_password_token=token,
        ).afirst()
        if not profile:
            await sync_to_async(messages.error)(
                request,
                "Invalid or expired token.",
            )
            return redirect("forgot-password")

        if profile.forget_password_token_expires_at and timezone.now() > (
            profile.forget_password_token_expires_at
        ):
            profile.forget_password_token = ""
            profile.forget_password_token_expires_at = None
            await profile.asave()
            await sync_to_async(messages.error)(
                request,
                "Invalid or expired token.",
            )
            return redirect("forgot-password")

        new_password = request.POST.get("password")
        confirm_password = request.POST.get("confirm-password")

        if not (new_password and confirm_password):
            await sync_to_async(messages.error)(
                request,
                "Please fill all fields.",
            )
            return await sync_to_async(render)(request, self.template_name)

        if new_password != confirm_password:
            await sync_to_async(messages.error)(
                request,
                "Passwords do not match.",
            )
            return await sync_to_async(render)(request, self.template_name)

        user = profile.user
        await sync_to_async(user.set_password)(new_password)
        await user.asave()

        profile.forget_password_token = ""
        profile.forget_password_token_expires_at = None
        await profile.asave()

        authenticated_user = await aauthenticate(
            request,
            username=user.username,
            password=new_password,
        )
        if authenticated_user:
            await alogin(request, authenticated_user)
            return redirect("index")
        await sync_to_async(messages.success)(
            request,
            "Password reset successful. Please log in.",
        )
        return redirect("login")
