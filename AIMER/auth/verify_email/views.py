from asgiref.sync import sync_to_async
from django.shortcuts import redirect
from django.contrib import messages
from django.conf import settings
from ..views import AuthView
from ..models import Profile
from ..helpers import send_verification_email
import uuid


class VerifyEmailTokenView(AuthView):
    async def get(self, request, token):
        try:
            profile = await Profile.objects.filter(email_token=token).afirst()
            profile.is_verified = True
            profile.email_token = ""
            await profile.asave()
            if not request.user.is_authenticated:
                # User is not already authenticated
                # Perform the email verification and any other necessary actions
                await sync_to_async(messages.success)(
                    request, "Email verified successfully"
                )
            return redirect("login")
            # Now, redirect to the login page

        except Profile.DoesNotExist:
            await sync_to_async(messages.error)(
                request, "Invalid token, please try again"
            )
            return redirect("verify-email-page")


class VerifyEmailView(AuthView):
    async def get(self, request):
        # Render the login page for users who are not logged in.
        return await sync_to_async(super().get)(request)


class SendVerificationView(AuthView):
    async def get(self, request):
        email, message = await self.get_email_and_message(request)

        if email:
            token = str(uuid.uuid4())
            user_profile = await Profile.objects.filter(email=email).afirst()
            user_profile.email_token = token
            await user_profile.asave()
            await send_verification_email(email, token)
            await sync_to_async(messages.success)(request, message)
        else:
            await sync_to_async(messages.error)(
                request, "Email not found in session"
            )

        return redirect("verify-email-page")

    async def get_email_and_message(self, request):
        if request.user.is_authenticated:
            email = await sync_to_async(lambda: request.user.profile.email)()

            if settings.EMAIL_HOST_USER and settings.EMAIL_HOST_PASSWORD:
                message = await sync_to_async(messages.success)(
                    request, "Verification email sent successfully"
                )
            else:
                message = await sync_to_async(messages.error)(
                    request,
                    "Email settings are not configured. Unable to send verification email.",
                )
        else:
            email = request.session.get("email")
            if settings.EMAIL_HOST_USER and settings.EMAIL_HOST_PASSWORD:
                message = (
                    "Resend verification email successfully" if email else None
                )
            else:
                message = await sync_to_async(messages.error)(
                    request,
                    "Email settings are not configured. Unable to send verification email.",
                )

        return email, message
