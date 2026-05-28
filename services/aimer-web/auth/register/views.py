# Copyright (c) 2026 AIMER contributors.
"""Registration views."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, override

from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.password_validation import validate_password
from django.contrib.auth.models import Group, User
from django.core.exceptions import ValidationError
from django.shortcuts import redirect
from django.utils import timezone

from auth.helpers import send_verification_email
from auth.models import Profile
from auth.security import audit_event
from auth.tokens import new_token_pair
from auth.views import AuthView

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse


DEFAULT_EMAIL_TOKEN_TTL_HOURS = 24


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
        username = (request.POST.get("username") or "").strip()
        email = (request.POST.get("email") or "").strip().lower()
        password = request.POST.get("password")

        if not (username and email and password):
            await sync_to_async(audit_event)(
                "auth.register.rejected",
                request=request,
                actor_identifier=email or username,
                metadata={"reason": "missing_fields"},
            )
            await sync_to_async(messages.error)(
                request,
                "Please fill in all required fields.",
            )
            return redirect("register")

        candidate_user = User(username=username, email=email)
        try:
            await sync_to_async(validate_password)(password, candidate_user)
        except ValidationError as exc:
            await sync_to_async(audit_event)(
                "auth.register.rejected",
                request=request,
                actor_identifier=email or username,
                metadata={"reason": "weak_password"},
            )
            await sync_to_async(messages.error)(request, " ".join(exc.messages))
            return redirect("register")

        if await User.objects.filter(username=username, email=email).aexists():
            await sync_to_async(audit_event)(
                "auth.register.rejected",
                request=request,
                actor_identifier=email,
                metadata={"reason": "existing_user"},
            )
            await sync_to_async(messages.error)(
                request,
                "Unable to create an account with those details.",
            )
            return redirect("register")
        if await User.objects.filter(email=email).aexists():
            await sync_to_async(audit_event)(
                "auth.register.rejected",
                request=request,
                actor_identifier=email,
                metadata={"reason": "existing_email"},
            )
            await sync_to_async(messages.error)(
                request,
                "Unable to create an account with those details.",
            )
            return redirect("register")
        if await User.objects.filter(username=username).aexists():
            await sync_to_async(audit_event)(
                "auth.register.rejected",
                request=request,
                actor_identifier=username,
                metadata={"reason": "existing_username"},
            )
            await sync_to_async(messages.error)(
                request,
                "Unable to create an account with those details.",
            )
            return redirect("register")

        created_user = await User.objects.acreate_user(
            username=username,
            email=email,
            password=password,
        )

        user_group, _created = await Group.objects.aget_or_create(name="client")
        await sync_to_async(created_user.groups.add)(user_group)

        token, token_hash = new_token_pair()
        token_ttl_hours = int(
            getattr(settings, "EMAIL_VERIFICATION_TOKEN_TTL_HOURS", 24)
            or DEFAULT_EMAIL_TOKEN_TTL_HOURS,
        )

        user_profile, _created = await Profile.objects.aget_or_create(user=created_user)
        user_profile.email_token = token_hash
        user_profile.email_token_expires_at = timezone.now() + timedelta(
            hours=token_ttl_hours,
        )
        user_profile.email = email
        await user_profile.asave()

        await send_verification_email(email, token)
        await sync_to_async(audit_event)(
            "auth.register.created",
            request=request,
            user=created_user,
            actor_identifier=email,
        )

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
