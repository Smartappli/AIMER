import uuid

from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.models import Group, User
from django.shortcuts import redirect

from ..helpers import send_verification_email
from ..models import Profile
from ..views import AuthView


class RegisterView(AuthView):
    async def get(self, request):
        if request.user.is_authenticated:
            # If the user is already logged in, redirect them to the home page or another appropriate page.
            return redirect(
                "index",
            )  # Replace 'index' with the actual URL name for the home page

        # Render the login page for users who are not logged in.
        return await sync_to_async(super().get)(request)

    async def post(self, request):
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")

        # Check if a user with the same username or email already exists
        if await User.objects.filter(username=username, email=email).aexists():
            await sync_to_async(messages.error)(
                request, "User already exists, Try logging in.",
            )
            return redirect("register")
        if await User.objects.filter(email=email).aexists():
            await sync_to_async(messages.error)(request, "Email already exists.")
            return redirect("register")
        if await User.objects.filter(username=username).aexists():
            await sync_to_async(messages.error)(request, "Username already exists.")
            return redirect("register")

        # Create the user and set their password
        created_user = await User.objects.acreate_user(
            username=username, email=email, password=password,
        )
        created_user.set_password(password)
        await created_user.asave()

        # Add the user to the 'client' group (or any other group you want to use as default for new users)
        user_group, created = await Group.objects.aget_or_create(name="client")
        await sync_to_async(created_user.groups.add)(user_group)

        # Generate a token and send a verification email here
        token = str(uuid.uuid4())

        # Set the token in the user's profile
        user_profile, created = await Profile.objects.aget_or_create(user=created_user)
        user_profile.email_token = token
        user_profile.email = email
        await user_profile.asave()

        await send_verification_email(email, token)

        if settings.EMAIL_HOST_USER and settings.EMAIL_HOST_PASSWORD:
            await sync_to_async(messages.success)(
                request, "Verification email sent successfully",
            )
        else:
            await sync_to_async(messages.error)(
                request,
                "Email settings are not configured. Unable to send verification email.",
            )

        request.session["email"] = email  ## Save email in session
        # Redirect to the verification page after successful registration
        return redirect("verify-email-page")
