# Copyright (c) 2026 AIMER contributors.
"""Login views."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import aauthenticate, alogin
from django.contrib.auth.models import User
from django.shortcuts import redirect
from django.utils.http import url_has_allowed_host_and_scheme

from auth.models import Profile
from auth.security import (
    audit_event,
    clear_login_failures,
    login_is_locked,
    record_login_failure,
)
from auth.views import AuthView

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse


GENERIC_LOGIN_ERROR = "Invalid credentials."
LOCKED_LOGIN_ERROR = "Too many failed attempts. Try again later."


class LoginView(AuthView):
    """Handle user login."""

    @override
    async def get(self, request: HttpRequest) -> HttpResponse:
        """
        Render login page for anonymous users.

        Returns:
            HttpResponse: The rendered login page or redirect.

        """
        is_auth = await sync_to_async(lambda: request.user.is_authenticated)()
        if is_auth:
            return redirect("index")
        return await sync_to_async(super().get)(request)

    @override
    async def post(self, request: HttpRequest) -> HttpResponse:
        """
        Authenticate and login a user.

        Returns:
            HttpResponse: Redirect to the requested page or index.

        """
        username = request.POST.get("email-username")
        password = request.POST.get("password")
        if not (username and password):
            await sync_to_async(audit_event)(
                "auth.login.missing_credentials",
                request=request,
                actor_identifier=str(username or ""),
            )
            await sync_to_async(messages.error)(
                request,
                "Please enter your username and password.",
            )
            return redirect("login")

        actor_identifier = str(username).strip().lower()
        if await sync_to_async(login_is_locked)(request, actor_identifier):
            await sync_to_async(audit_event)(
                "auth.login.locked",
                request=request,
                actor_identifier=actor_identifier,
            )
            await sync_to_async(messages.error)(request, LOCKED_LOGIN_ERROR)
            return redirect("login")

        if "@" in username:
            user_by_email = await User.objects.filter(email=username).afirst()
            if user_by_email is None:
                await sync_to_async(record_login_failure)(
                    request,
                    actor_identifier,
                )
                await sync_to_async(audit_event)(
                    "auth.login.failed",
                    request=request,
                    actor_identifier=actor_identifier,
                    metadata={"reason": "unknown_email"},
                )
                await sync_to_async(messages.error)(
                    request,
                    GENERIC_LOGIN_ERROR,
                )
                return redirect("login")
            username = user_by_email.username

        user_by_username = await User.objects.filter(username=username).afirst()
        if user_by_username is None:
            await sync_to_async(record_login_failure)(
                request,
                actor_identifier,
            )
            await sync_to_async(audit_event)(
                "auth.login.failed",
                request=request,
                actor_identifier=actor_identifier,
                metadata={"reason": "unknown_username"},
            )
            await sync_to_async(messages.error)(
                request,
                GENERIC_LOGIN_ERROR,
            )
            return redirect("login")

        authenticated_user = await aauthenticate(
            request,
            username=username,
            password=password,
        )
        if authenticated_user is None:
            locked = await sync_to_async(record_login_failure)(
                request,
                actor_identifier,
            )
            await sync_to_async(audit_event)(
                "auth.login.failed",
                request=request,
                user=user_by_username,
                actor_identifier=actor_identifier,
                metadata={"reason": "bad_password", "locked": locked},
            )
            await sync_to_async(messages.error)(
                request,
                LOCKED_LOGIN_ERROR if locked else GENERIC_LOGIN_ERROR,
            )
            return redirect("login")

        if await self.email_verification_blocks_login(request, authenticated_user):
            await sync_to_async(audit_event)(
                "auth.login.blocked_unverified_email",
                request=request,
                user=authenticated_user,
                actor_identifier=actor_identifier,
            )
            return redirect("verify-email-page")

        await sync_to_async(clear_login_failures)(request, actor_identifier)
        await alogin(request, authenticated_user)
        await sync_to_async(audit_event)(
            "auth.login.success",
            request=request,
            user=authenticated_user,
            actor_identifier=actor_identifier,
        )
        next_url = request.POST.get("next", "")
        if next_url and url_has_allowed_host_and_scheme(
            url=next_url,
            allowed_hosts={request.get_host(), *getattr(settings, "ALLOWED_HOSTS", [])},
            require_https=not settings.DEBUG,
        ):
            return redirect(next_url)
        return redirect("index")

    @staticmethod
    async def email_verification_blocks_login(request: HttpRequest, user: User) -> bool:
        """Return whether configured email verification should block login."""
        if not getattr(settings, "EMAIL_VERIFICATION_REQUIRED", False):
            return False
        if user.is_staff or user.is_superuser:
            return False

        profile = await Profile.objects.filter(user=user).afirst()
        if profile and profile.is_verified:
            return False

        request.session["email"] = user.email
        await sync_to_async(messages.error)(
            request,
            "Please verify your email before logging in.",
        )
        return True
