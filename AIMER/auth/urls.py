from django.urls import path
from django.contrib.auth.views import LogoutView
from .register.views import RegisterView
from .login.views import LoginView
from .forgot_password.views import ForgetPasswordView
from .reset_password.views import ResetPasswordView
from .verify_email.views import  VerifyEmailTokenView , VerifyEmailView, SendVerificationView


urlpatterns = [
    path(
        "login/",
        LoginView.as_view(template_name="auth/login.html"),
        name="login",
    ),

    path(
        "logout/",
        LogoutView.as_view(),
        name="logout",
    ),

    path(
        "register/",
        RegisterView.as_view(template_name="auth/register.html"),
        name="register",
    ),

    path(
        "verify_email/",
        VerifyEmailView.as_view(template_name="auth/verify_email.html"),
        name="verify-email-page",
    ),

    path(
        "verify/email/<str:token>/",
        VerifyEmailTokenView.as_view(),
        name="verify-email",
    ),

    path(
        "send_verification/",
        SendVerificationView.as_view(),
        name="send-verification",
    ),

    path(
        "forgot_password/",
        ForgetPasswordView.as_view(template_name="auth/forgot_password.html"),
        name="forgot-password",
    ),

    path(
        "reset_password/<str:token>/",
        ResetPasswordView.as_view(template_name="auth/reset_password.html"),
        name="reset-password",
    ),

]
