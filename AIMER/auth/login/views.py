from asgiref.sync import sync_to_async
from django.shortcuts import redirect
from django.contrib.auth import aauthenticate, alogin
from django.contrib.auth.models import User
from django.contrib import messages
from ..views import AuthView


class LoginView(AuthView):
    async def get(self, request):
        is_auth = await sync_to_async(lambda: request.user.is_authenticated)()
        if is_auth:
            # If the user is already logged in, redirect them to the home page or another appropriate page.
            return redirect("index")  # Replace 'index' with the actual URL name for the home page

        # Render the login page for users who are not logged in.
        return await sync_to_async(super().get)(request)

    async def post(self, request):
        if request.method == "POST":
            username = request.POST.get("email-username")
            password = request.POST.get("password")

            if not (username and password):
                await sync_to_async(messages.error)(request, "Please enter your username and password.")
                return redirect("login")

            if "@" in username:
                user_email = await User.objects.filter(email=username).afirst()
                if user_email is None:
                    await sync_to_async(messages.error)(request, "Please enter a valid email.")
                    return redirect("login")
                username = user_email.username

            user_email = await User.objects.filter(username=username).afirst()
            if user_email is None:
                await sync_to_async(messages.error)(request, "Please enter a valid username.")
                return redirect("login")

            authenticated_user = await aauthenticate(request, username=username, password=password)
            if authenticated_user is not None:
                # Login the user if authentication is successful
                await alogin(request, authenticated_user)

                # Redirect to the page the user was trying to access before logging in
                if "next" in request.POST:
                    return redirect(request.POST["next"])
                else: # Redirect to the home page or another appropriate page
                    return redirect("index")
            else:
                await sync_to_async(messages.error)(request, "Please enter a valid username.")
                return redirect("login")
