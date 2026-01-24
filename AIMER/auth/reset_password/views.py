from asgiref.sync import sync_to_async
from django.contrib import messages
from django.contrib.auth import aauthenticate, alogin
from django.shortcuts import redirect, render

from ..models import Profile
from ..views import AuthView


class ResetPasswordView(AuthView):
    async def get(self, request, token):
        if request.user.is_authenticated:
            # If the user is already logged in, redirect them to the home page or another appropriate page.
            return redirect(
                "index",
            )  # Replace 'index' with the actual URL name for the home page

        # Render the login page for users who are not logged in.
        return await sync_to_async(super().get)(request)

    async def post(self, request, token):
        try:
            profile = await Profile.objects.aget(forget_password_token=token)
        except Profile.DoesNotExist:
            await sync_to_async(messages.error)(
                request, "Invalid or expired token.",
            )
            return redirect("forgot-password")

        if request.method == "POST":
            new_password = request.POST.get("password")
            confirm_password = request.POST.get("confirm-password")

            if not (new_password and confirm_password):
                await sync_to_async(messages.error)(
                    request, "Please fill all fields.",
                )
                return await sync_to_async(render)(
                    request, "reset-password.html",
                )

            if new_password != confirm_password:
                await sync_to_async(messages.error)(
                    request, "Passwords do not match.",
                )
                return await sync_to_async(render)(
                    request, "reset-password.html",
                )

            user = profile.user
            await sync_to_async(user.set_password)(new_password)
            await user.asave()

            # Clear the forget_password_token
            profile.forget_password_token = ""
            await profile.asave()

            # Log the user in after a successful password reset
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
