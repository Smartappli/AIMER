"""Security-focused tests for custom authentication views."""

from __future__ import annotations

from datetime import timedelta

from auth.models import Profile
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone


class AuthSecurityTests(TestCase):
    """Production auth hardening checks."""

    def test_register_rejects_weak_password(self) -> None:
        """Weak passwords must not create accounts."""
        response = self.client.post(
            reverse("register"),
            {
                "username": "weak-user",
                "email": "weak-user@example.com",
                "password": "123",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertFalse(
            get_user_model().objects.filter(username="weak-user").exists(),
        )

    def test_reset_password_rejects_weak_password(self) -> None:
        """Password reset must use Django password validators."""
        user = get_user_model().objects.create_user(
            username="reset-user",
            email="reset-user@example.com",
            password="Original-Strong-Passphrase-2026",
        )
        profile = Profile.objects.get(user=user)
        profile.forget_password_token = "reset-token"
        profile.forget_password_token_expires_at = timezone.now() + timedelta(hours=1)
        profile.save()

        response = self.client.post(
            reverse("reset-password", kwargs={"token": "reset-token"}),
            {
                "password": "123",
                "confirm-password": "123",
            },
        )

        self.assertEqual(response.status_code, 200)
        user.refresh_from_db()
        profile.refresh_from_db()
        self.assertTrue(user.check_password("Original-Strong-Passphrase-2026"))
        self.assertEqual(profile.forget_password_token, "reset-token")

    @override_settings(EMAIL_VERIFICATION_REQUIRED=True)
    def test_login_blocks_unverified_user_when_required(self) -> None:
        """Unverified users must not receive a session when verification is required."""
        get_user_model().objects.create_user(
            username="unverified",
            email="unverified@example.com",
            password="Strong-Passphrase-2026",
        )

        response = self.client.post(
            reverse("login"),
            {
                "email-username": "unverified",
                "password": "Strong-Passphrase-2026",
            },
        )

        self.assertRedirects(response, reverse("verify-email-page"))
        self.assertNotIn("_auth_user_id", self.client.session)

    @override_settings(EMAIL_VERIFICATION_REQUIRED=True)
    def test_login_allows_verified_user_when_required(self) -> None:
        """Verified users can log in when verification is required."""
        user = get_user_model().objects.create_user(
            username="verified",
            email="verified@example.com",
            password="Strong-Passphrase-2026",
        )
        profile = Profile.objects.get(user=user)
        profile.is_verified = True
        profile.save()

        response = self.client.post(
            reverse("login"),
            {
                "email-username": "verified",
                "password": "Strong-Passphrase-2026",
            },
        )

        self.assertRedirects(response, reverse("index"))
        self.assertEqual(int(self.client.session["_auth_user_id"]), user.pk)

    def test_send_verification_requires_post(self) -> None:
        """Resending verification email is a CSRF-protected POST action."""
        response = self.client.get(reverse("send-verification"))

        self.assertEqual(response.status_code, 405)
