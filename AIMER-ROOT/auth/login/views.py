# Copyright (c) 2026 AIMER contributors.
"""Login views."""

from __future__ import annotations

from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import aauthenticate, alogin
from django.contrib.auth.models import User
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect
from django.utils.http import url_has_allowed_host_and_scheme

from auth.views import AuthView


class LoginView(AuthView):
    """Handle user login."""

    async def get(self, request: HttpRequest) -> HttpResponse:
        """Render login page for anonymous users."""
        is_auth = await sync_to_async(lambda: request.user.is_authenticated)()
        if is_auth:
            return redirect("index")
        return await sync_to_async(super().get)(request)

    async def post(self, request: HttpRequest) -> HttpResponse:
        """Authenticate and login a user."""
        username = request.POST.get("email-username")
        password = request.POST.get("password")
        if not (username and password):
            await sync_to_async(messages.error)(
                request,
                "Please enter your username and password.",
            )
            return redirect("login")

        if "@" in username:
            user_by_email = await User.objects.filter(email=username).afirst()
            if user_by_email is None:
                await sync_to_async(messages.error)(
                    request,
                    "Please enter a valid email.",
                )
                return redirect("login")
            username = user_by_email.username

        user_by_username = await User.objects.filter(username=username).afirst()
        if user_by_username is None:
            await sync_to_async(messages.error)(
                request,
                "Please enter a valid username.",
            )
            return redirect("login")

        authenticated_user = await aauthenticate(
            request,
            username=username,
            password=password,
        )
        if authenticated_user is None:
            await sync_to_async(messages.error)(
                request,
                "Please enter a valid username.",
            )
            return redirect("login")

        await alogin(request, authenticated_user)
        next_url = request.POST.get("next", "")
        if next_url and url_has_allowed_host_and_scheme(
            url=next_url,
            allowed_hosts={request.get_host(), *getattr(settings, "ALLOWED_HOSTS", [])},
            require_https=not settings.DEBUG,
        ):
            return redirect(next_url)
        return redirect("index")
