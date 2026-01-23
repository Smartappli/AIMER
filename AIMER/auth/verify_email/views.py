import uuid

from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib import messages
from django.shortcuts import redirect

from ..helpers import send_verification_email
from ..models import Profile
from ..views import AuthView


class VerifyEmailTokenView(AuthView):
    """Verify a user's email address using a token.

    GET /verify-email/<token>:
    - Finds the Profile with `email_token == token`.
    - Sets `is_verified = True` and clears `email_token`.
    - Shows a success message (mainly for non-authenticated users).
    - Redirects to the login page.

    Error handling:
    - If the token is invalid, shows an error message and redirects to the
      verify email page.
    """

    async def get(self, request, token):
        """Handle token verification.

        Args:
            request: Django HttpRequest.
            token: Verification token extracted from the URL.

        Returns:
            An HTTP redirect response.

        """
        try:
            profile = await Profile.objects.filter(email_token=token).afirst()
            profile.is_verified = True
            profile.email_token = ""
            await profile.asave()
            if not request.user.is_authenticated:
                # User is not already authenticated
                # Perform the email verification and any other necessary actions
                await sync_to_async(messages.success)(
                    request,
                    "Email verified successfully",
                )
            return redirect("login")
            # Now, redirect to the login page

        except Profile.DoesNotExist:
            await sync_to_async(messages.error)(
                request,
                "Invalid token, please try again",
            )
            return redirect("verify-email-page")


class VerifyEmailView(AuthView):
    """Display the verify email page.

    This is typically a page where users are informed that they must verify their
    email address and where they can trigger a resend of the verification email.
    """

    async def get(self, request):
        """Render the verify email page.

        Args:
            request: Django HttpRequest.

        Returns:
            The response returned by the parent AuthView's GET handler.

        """
        return await sync_to_async(super().get)(request)


class SendVerificationView(AuthView):
    """Generate and send a verification email.

    GET /send-verification:
    - Determines the target email (authenticated user profile or session).
    - Generates a new UUID token, saves it into Profile.email_token.
    - Sends the verification email.
    - Displays a success or error message.
    - Redirects back to the verify email page.
    """

    async def get(self, request):
        """Send a (re)verification email to the user.

        Args:
            request: Django HttpRequest.

        Returns:
            An HTTP redirect response to the verify email page.

        """
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
        """Resolve the recipient email and the user-facing message to display.

        Rules:
        - If authenticated, use `request.user.profile.email`.
        - If not authenticated, use `request.session["email"]` if present.
        - If EMAIL settings are missing, prepare an error message.

        Args:
            request: Django HttpRequest.

        Returns:
            A tuple (email, message) where:
            - email is a string or None
            - message is either a string or a messages.* result depending on path

        """
        if request.user.is_authenticated:
            email = await sync_to_async(lambda: request.user.profile.email)()

            if settings.EMAIL_HOST_USER and settings.EMAIL_HOST_PASSWORD:
                message = await sync_to_async(messages.success)(
                    request,
                    "Verification email sent successfully",
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
