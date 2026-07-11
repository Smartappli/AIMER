"""Security-focused tests for custom authentication views."""

from __future__ import annotations

import json
from datetime import timedelta
from unittest.mock import AsyncMock, patch

from auth.middleware import AdminAuditMiddleware
from auth.models import Profile, SecurityAuditEvent
from auth.security import audit_event, consume_email_action
from auth.tokens import hash_token
from django.contrib.auth import get_user_model
from django.http import HttpResponse
from django.test import RequestFactory
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
        profile.forget_password_token = hash_token("reset-token")
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
        self.assertEqual(profile.forget_password_token, hash_token("reset-token"))

    def test_reset_password_get_accepts_url_token(self) -> None:
        """The password-reset link must render instead of failing on its token."""
        response = self.client.get(
            reverse("reset-password", kwargs={"token": "reset-token"}),
        )

        self.assertEqual(response.status_code, 200)

    @override_settings(
        AUTH_EMAIL_ACTION_LIMIT=1,
        AUTH_EMAIL_ACTION_WINDOW_SECONDS=60,
    )
    def test_email_actions_are_rate_limited_per_ip_and_recipient(self) -> None:
        """Repeated delivery requests must be throttled without global lockout."""
        request = RequestFactory().post(
            "/forgot_password/",
            REMOTE_ADDR="203.0.113.20",
        )

        self.assertFalse(
            consume_email_action(request, "password_reset", "user@example.org"),
        )
        self.assertTrue(
            consume_email_action(request, "password_reset", "user@example.org"),
        )
        self.assertFalse(
            consume_email_action(request, "password_reset", "other@example.org"),
        )

    @override_settings(
        AUTH_EMAIL_ACTION_LIMIT=1,
        AUTH_EMAIL_ACTION_WINDOW_SECONDS=60,
        EMAIL_HOST_USER="smtp-user",
        EMAIL_HOST_PASSWORD="smtp-password",
    )
    @patch(
        "auth.forgot_password.views.send_password_reset_email",
        new_callable=AsyncMock,
    )
    def test_password_reset_delivery_is_rate_limited(self, send_email) -> None:
        """The endpoint must not repeatedly send reset links to one recipient."""
        user = get_user_model().objects.create_user(
            username="limited-reset-user",
            email="limited-reset-user@example.com",
            password="Strong-Passphrase-2026",
        )

        first = self.client.post(reverse("forgot-password"), {"email": user.email})
        second = self.client.post(reverse("forgot-password"), {"email": user.email})

        self.assertEqual(first.status_code, 302)
        self.assertEqual(second.status_code, 302)
        send_email.assert_awaited_once()
        self.assertTrue(
            SecurityAuditEvent.objects.filter(
                event_type="auth.password_reset.rate_limited",
            ).exists(),
        )

    def test_password_reset_token_is_hashed_at_rest(self) -> None:
        """Password reset requests must not persist the raw email token."""
        user = get_user_model().objects.create_user(
            username="token-user",
            email="token-user@example.com",
            password="Original-Strong-Passphrase-2026",
        )

        response = self.client.post(
            reverse("forgot-password"),
            {"email": user.email},
        )

        self.assertEqual(response.status_code, 302)
        profile = Profile.objects.get(user=user)
        self.assertIsNotNone(profile.forget_password_token)
        self.assertEqual(len(profile.forget_password_token), 64)
        self.assertNotEqual(profile.forget_password_token, "reset-token")

    def test_failed_login_is_audited(self) -> None:
        """Failed authentication attempts must produce security audit events."""
        get_user_model().objects.create_user(
            username="audit-login",
            email="audit-login@example.com",
            password="Strong-Passphrase-2026",
        )

        response = self.client.post(
            reverse("login"),
            {
                "email-username": "audit-login",
                "password": "wrong-password",
            },
        )

        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            SecurityAuditEvent.objects.filter(event_type="auth.login.failed").exists(),
        )

    def test_audit_event_emits_structured_redacted_log(self) -> None:
        """Audit events must produce SIEM-collectable structured logs."""
        request = RequestFactory().get("/login/", HTTP_USER_AGENT="test-agent")

        with self.assertLogs("aimer.security.audit", level="INFO") as captured:
            audit_event(
                "auth.test",
                request=request,
                actor_identifier="alice@example.org",
                metadata={
                    "reason": "unit_test",
                    "reset_token": "raw-token-value",
                },
            )

        payload = json.loads(captured.output[0].split("INFO:aimer.security.audit:")[1])
        self.assertEqual(payload["event_type"], "auth.test")
        self.assertEqual(payload["metadata"]["reason"], "unit_test")
        self.assertEqual(payload["metadata"]["reset_token"], "[REDACTED]")
        self.assertTrue(payload["persisted"])

    def test_admin_mutations_are_audited(self) -> None:
        """Mutating staff admin requests must produce privileged audit events."""
        staff_user = get_user_model().objects.create_user(
            username="admin-auditor",
            email="admin-auditor@example.com",
            password="Strong-Passphrase-2026",
            is_staff=True,
        )
        request = RequestFactory().post("/admin/auth/user/1/change/")
        request.user = staff_user
        middleware = AdminAuditMiddleware(lambda _request: HttpResponse(status=302))

        response = middleware(request)

        self.assertEqual(response.status_code, 302)
        event = SecurityAuditEvent.objects.get(event_type="admin.privileged_action")
        self.assertEqual(event.user, staff_user)
        self.assertEqual(event.metadata["method"], "POST")
        self.assertEqual(event.metadata["status_code"], 302)

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
