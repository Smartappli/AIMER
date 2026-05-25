# Copyright (c) 2026 AIMER contributors.
"""Registration views."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, override

from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.models import Group, User
from django.shortcuts import redirect

from auth.helpers import send_verification_email
from auth.models import Profile
from auth.views import AuthView

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse


class RegisterView(AuthView):
    """Register a new user and send email verification."""

    @override
    async def get(self, request: HttpRequest) -> HttpResponse:
        """
        Render registration form for anonymous users.

        Returns:
            HttpResponse: Registration page or redirect for authenticated users.

        """
        if request.user.is_authenticated:
            return redirect("index")

        return await sync_to_async(super().get)(request)

    @override
    async def post(self, request: HttpRequest) -> HttpResponse:
        """
        Create an account and trigger verification email workflow.

        Returns:
            HttpResponse: Redirect to verification page or back to register page.

        """
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")

        if not (username and email and password):
            await sync_to_async(messages.error)(
                request,
                "Please fill in all required fields.",
            )
            return redirect("register")

        if await User.objects.filter(username=username, email=email).aexists():
            await sync_to_async(messages.error)(
                request,
                "User already exists, Try logging in.",
            )
            return redirect("register")
        if await User.objects.filter(email=email).aexists():
            await sync_to_async(messages.error)(
                request,
                "Email already exists.",
            )
            return redirect("register")
        if await User.objects.filter(username=username).aexists():
            await sync_to_async(messages.error)(
                request,
                "Username already exists.",
            )
            return redirect("register")

        created_user = await User.objects.acreate_user(
            username=username,
            email=email,
            password=password,
        )

        user_group, _created = await Group.objects.aget_or_create(name="client")
        await sync_to_async(created_user.groups.add)(user_group)

        token = str(uuid.uuid4())

        user_profile, _created = await Profile.objects.aget_or_create(user=created_user)
        user_profile.email_token = token
        user_profile.email = email
        await user_profile.asave()

        await send_verification_email(email, token)

        if settings.EMAIL_HOST_USER and settings.EMAIL_HOST_PASSWORD:
            await sync_to_async(messages.success)(
                request,
                "Verification email sent successfully",
            )
        else:
            await sync_to_async(messages.error)(
                request,
                "Email settings are not configured. Unable to send verification email.",
            )

        request.session["email"] = email
        return redirect("verify-email-page")
