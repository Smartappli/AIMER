import uuid
from datetime import timedelta

from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.models import User
from django.shortcuts import redirect
from django.utils import timezone

from ..helpers import send_password_reset_email
from ..models import Profile
from ..views import AuthView


class ForgetPasswordView(AuthView):
    async def get(self, request):
        if request.user.is_authenticated:
            # If the user is already logged in, redirect them to the home page or another appropriate page.
            return redirect(
                "index",
            )  # Replace 'index' with the actual URL name for the home page

        # Render the login page for users who are not logged in.
        return await sync_to_async(super().get)(request)

    async def post(self, request):
        if request.method == "POST":
            email = request.POST.get("email")
            if not email:
                await sync_to_async(messages.error)(
                    request,
                    "Please enter your email address.",
                )
                return redirect("forgot-password")

            user = await User.objects.filter(email=email).afirst()

            # Generate a token and send a password reset email here
            token = str(uuid.uuid4())

            # Set the token in the user's profile and add an expiration time (e.g., 24 hours from now)
            expiration_time = timezone.now() + timedelta(hours=24)

            if user:
                user_profile, created = await Profile.objects.aget_or_create(
                    user=user,
                )
                user_profile.forget_password_token = token
                user_profile.forget_password_token_expires_at = expiration_time
                await user_profile.asave()

                # Send the password reset email
                await send_password_reset_email(email, token)

            if settings.EMAIL_HOST_USER and settings.EMAIL_HOST_PASSWORD:
                await sync_to_async(messages.success)(
                    request,
                    "If an account exists for that email, a reset link has been sent.",
                )
            else:
                await sync_to_async(messages.error)(
                    request,
                    "Email settings are not configured. Unable to send verification email.",
                )

            return redirect("forgot-password")
