# Copyright (c) 2026 AIMER contributors.
"""Unit tests for the Giovani voice-assistant integration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from django.contrib.auth import get_user_model
from django.contrib.staticfiles import finders
from django.test import TestCase

HTTP_OK = 200


class VoiceAssistantUnitTests(TestCase):
    """Verify the rendered assistant and its server-side language contract."""

    def setUp(self) -> None:
        """Authenticate a fresh dashboard user."""
        user = get_user_model().objects.create_user(
            username=f"giovani-unit-{uuid4().hex}",
            email=f"giovani-unit-{uuid4().hex}@example.com",
            password=uuid4().hex,
        )
        self.client.force_login(user)

    def _check(self, condition: object, message: str = "Expectation failed") -> None:
        """Fail with a readable message when a condition is false."""
        if not bool(condition):
            self.fail(message)

    def _check_equal(self, left: object, right: object) -> None:
        """Fail when two values differ."""
        if left != right:
            self.fail(f"Values differ: {left!r} != {right!r}")

    @patch("website.views._discover_scientific_articles", return_value=[])
    def test_dashboard_renders_giovani_avatar_and_locales(
        self,
        discover: Mock,
    ) -> None:
        """The dashboard exposes Giovani's portrait, rig, and four locales."""
        response = self.client.get("/dashboard/")

        self._check_equal(response.status_code, HTTP_OK)
        discover.assert_called_once()
        self.assertContains(response, "Giovani")
        self.assertContains(response, "giovani-assistant.jpg")
        self.assertContains(response, 'id="voice-assistant-avatar"')
        self.assertContains(response, "voice-avatar__mouth-mask")
        self.assertContains(response, 'id="voice-assistant-rate"')
        self.assertContains(response, "voice-assistant__attitude-cue--listening")
        self.assertContains(response, "voice-assistant__attitude-cue--thinking")
        self.assertContains(response, "voice-assistant__attitude-cue--speaking")
        self.assertContains(response, "voice-assistant__attitude-cue--success")
        self.assertContains(response, "voice-assistant__attitude-cue--error")
        self.assertContains(response, 'id="voice-assistant-demo"')
        self.assertContains(response, 'data-demo-step="dictation"')
        self.assertContains(response, 'data-demo-step="analysis"')
        self.assertContains(response, 'data-demo-step="response"')
        for locale in ("fr-FR", "en-US", "nl-NL", "de-DE"):
            with self.subTest(locale=locale):
                self.assertContains(response, f'value="{locale}"')
        for rate in ("0.75", "1", "1.25", "1.5"):
            with self.subTest(rate=rate):
                self.assertContains(response, f'value="{rate}"')

    def test_giovani_portrait_is_a_nonempty_jpeg_static_asset(self) -> None:
        """The portrait referenced by the template is collected as a valid JPEG."""
        located_asset = finders.find("img/avatars/giovani-assistant.jpg")

        self._check(isinstance(located_asset, str), "Giovani portrait was not found")
        asset = Path(located_asset)
        self._check(asset.stat().st_size > 0, "Giovani portrait is empty")
        with asset.open("rb") as image:
            self._check_equal(image.read(3), b"\xff\xd8\xff")

    @patch("website.views.recommend_models")
    def test_api_normalizes_and_forwards_assistant_language(
        self,
        recommend: Mock,
    ) -> None:
        """The Django API forwards a regional locale as its supported base code."""
        recommend.return_value = {
            "query": "Klassifikation von MRT-Bildern",
            "language": "de",
            "query_profile": {"tasks": ["classification"], "modalities": ["mri"]},
            "recommended_models": [],
            "safety_notice": "Nur zur experimentellen Unterstützung.",
        }

        response = self.client.get(
            "/api/rag/recommend/",
            {
                "q": "Klassifikation von MRT-Bildern",
                "language": "de-DE",
            },
        )

        self._check_equal(response.status_code, HTTP_OK)
        self._check_equal(response.json()["language"], "de")
        self._check_equal(recommend.call_args.kwargs["language"], "de")
        self._check(recommend.call_args.kwargs["strict_openrag"])
