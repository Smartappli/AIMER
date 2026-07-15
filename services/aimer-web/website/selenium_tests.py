# Copyright (c) 2026 AIMER contributors.
"""
Selenium browser tests for the Giovani voice assistant.

Run with:
    python manage.py test website.selenium_tests
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, ClassVar
from unittest import SkipTest
from unittest.mock import patch
from uuid import uuid4

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.test import override_settings
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait

if TYPE_CHECKING:
    from selenium.webdriver.remote.webdriver import WebDriver

WAIT_SECONDS = 10
EXPECTED_VISEME_COUNT = 5
MIN_MOUTH_OPACITY = 0.5

FAKE_BROWSER_APIS = r"""
class FakeSpeechSynthesisUtterance extends EventTarget {
  constructor(text) {
    super();
    this.text = text;
    this.lang = '';
    this.rate = 1;
    this.voice = null;
  }
}

class FakeSpeechRecognition extends EventTarget {
  constructor() {
    super();
    this.lang = '';
    this.continuous = false;
    this.interimResults = true;
  }

  start() {
    window.__recognitionLanguage = this.lang;
    this.dispatchEvent(new Event('start'));
    window.setTimeout(() => {
      const result = new Event('result');
      Object.defineProperties(result, {
        resultIndex: {value: 0},
        results: {value: [[{transcript: 'classificatie MRI'}]]}
      });
      this.dispatchEvent(result);
    }, 30);
    window.setTimeout(() => this.dispatchEvent(new Event('end')), 250);
  }

  stop() {
    this.dispatchEvent(new Event('end'));
  }

  abort() {
    this.dispatchEvent(new Event('end'));
  }
}

const fakeSpeechSynthesis = {
  cancel() {},
  getVoices() {
    return ['fr-FR', 'en-US', 'nl-NL', 'de-DE'].map(lang => ({lang}));
  },
  speak(utterance) {
    window.__speechRate = utterance.rate;
    window.__spokenText = utterance.text;
    window.setTimeout(() => utterance.dispatchEvent(new Event('start')), 10);
    window.setTimeout(() => {
      const boundary = new Event('boundary');
      Object.defineProperty(boundary, 'charIndex', {value: 2});
      Object.defineProperty(boundary, 'charLength', {value: 4});
      utterance.dispatchEvent(boundary);
    }, 80);
    window.setTimeout(() => utterance.dispatchEvent(new Event('end')), 900);
  }
};

Object.defineProperty(window, 'SpeechSynthesisUtterance', {
  configurable: true,
  value: FakeSpeechSynthesisUtterance
});
Object.defineProperty(window, 'speechSynthesis', {
  configurable: true,
  value: fakeSpeechSynthesis
});
Object.defineProperty(window, 'SpeechRecognition', {
  configurable: true,
  value: FakeSpeechRecognition
});

window.fetch = async () => {
  await new Promise(resolve => window.setTimeout(resolve, 180));
  return new Response(JSON.stringify({
    query: 'classification MRI',
    language: 'fr',
    query_profile: {tasks: ['classification'], modalities: ['mri']},
    recommended_models: [{
      model_name: 'ResNet',
      confidence: 0.91,
      rationale: 'Recommandation de test Selenium.',
      evidence: []
    }],
    safety_notice: 'Validation clinique requise.'
  }), {
    status: 200,
    headers: {'Content-Type': 'application/json'}
  });
};
"""


@override_settings(
    ALLOWED_HOSTS=["localhost", "127.0.0.1"],
    SECURE_SSL_REDIRECT=False,
    SESSION_COOKIE_SECURE=False,
)
class VoiceAssistantSeleniumTests(StaticLiveServerTestCase):
    """Exercise Giovani in a real headless browser."""

    browser: ClassVar[WebDriver]

    @classmethod
    def setUpClass(cls) -> None:
        """
        Start Chrome and install deterministic browser API doubles.

        Raises:
            SkipTest: When Chrome or its WebDriver cannot be started.
            WebDriverException: When Chrome cannot be started in CI.

        """
        super().setUpClass()
        options = Options()
        for argument in (
            "--headless=new",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--window-size=1440,1200",
        ):
            options.add_argument(argument)
        try:
            cls.browser = webdriver.Chrome(options=options)
        except WebDriverException as exc:
            super().tearDownClass()
            if os.environ.get("CI"):
                raise
            msg = f"Chrome WebDriver is unavailable: {exc}"
            raise SkipTest(msg) from exc
        cls.browser.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": FAKE_BROWSER_APIS},
        )

    def _check(self, condition: object, message: str = "Expectation failed") -> None:
        """Fail with a readable message when a browser condition is false."""
        if not bool(condition):
            self.fail(message)

    def _check_equal(self, left: object, right: object) -> None:
        """Fail when two browser-observed values differ."""
        if left != right:
            self.fail(f"Values differ: {left!r} != {right!r}")

    @classmethod
    def tearDownClass(cls) -> None:
        """Stop Chrome after the live-server test class."""
        if hasattr(cls, "browser"):
            cls.browser.quit()
        super().tearDownClass()

    def setUp(self) -> None:
        """Create an authenticated browser session for each test."""
        user = get_user_model().objects.create_user(
            username=f"giovani-selenium-{uuid4().hex}",
            email=f"giovani-selenium-{uuid4().hex}@example.com",
            password=uuid4().hex,
        )
        self.client.force_login(user)
        self.browser.get(f"{self.live_server_url}/healthz/")
        self.browser.delete_all_cookies()
        self.browser.execute_script("window.localStorage.clear();")
        session_cookie = self.client.cookies[settings.SESSION_COOKIE_NAME]
        self.browser.add_cookie(
            {
                "name": settings.SESSION_COOKIE_NAME,
                "value": session_cookie.value,
                "path": "/",
            },
        )
        self.wait = WebDriverWait(self.browser, WAIT_SECONDS)

    def _open_dashboard(self) -> None:
        """Open the authenticated dashboard without scanning the real corpus."""
        with patch(
            "website.views._discover_scientific_articles",
            return_value=[],
        ):
            self.browser.get(f"{self.live_server_url}/dashboard/")
        self.wait.until(
            lambda driver: driver.find_element(By.ID, "voice-assistant-avatar"),
        )

    def test_dashboard_renders_giovani_and_all_language_options(self) -> None:
        """Giovani's portrait, lip rig, and locales render in the browser."""
        self._open_dashboard()

        self._check("Giovani" in self.browser.page_source)
        portrait = self.browser.find_element(By.CSS_SELECTOR, ".voice-assistant__photo")
        portrait_source = portrait.get_attribute("src") or ""
        self._check(portrait_source.endswith("giovani-assistant.jpg"))
        self._check_equal(
            len(self.browser.find_elements(By.CSS_SELECTOR, ".voice-avatar__viseme")),
            EXPECTED_VISEME_COUNT,
        )
        language = Select(self.browser.find_element(By.ID, "voice-assistant-language"))
        self._check_equal(
            [option.get_attribute("value") for option in language.options],
            ["fr-FR", "en-US", "nl-NL", "de-DE"],
        )

    def test_language_selection_updates_and_persists_giovani_ui(self) -> None:
        """Changing language translates Giovani and survives a page reload."""
        self._open_dashboard()
        language_element = self.browser.find_element(By.ID, "voice-assistant-language")
        Select(language_element).select_by_value("nl-NL")

        greeting = self.browser.find_element(
            By.CSS_SELECTOR, '[data-i18n="greetingTitle"]'
        )
        self.wait.until(lambda _driver: "ik ben Giovani" in greeting.text)
        self._check_equal(
            self.browser.execute_script(
                "return window.localStorage.getItem('aimer-assistant-language');",
            ),
            "nl-NL",
        )

        with patch("website.views._discover_scientific_articles", return_value=[]):
            self.browser.refresh()
        refreshed_language = Select(
            self.wait.until(
                lambda driver: driver.find_element(By.ID, "voice-assistant-language"),
            ),
        )
        self._check_equal(
            refreshed_language.first_selected_option.get_attribute("value"),
            "nl-NL",
        )

    def test_speech_recognition_uses_selected_language(self) -> None:
        """The speech recognizer receives NL and writes its transcript to the form."""
        self._open_dashboard()
        Select(
            self.browser.find_element(By.ID, "voice-assistant-language")
        ).select_by_value(
            "nl-NL",
        )
        self.browser.find_element(By.ID, "voice-assistant-microphone").click()

        avatar = self.browser.find_element(By.ID, "voice-assistant-avatar")
        self.wait.until(
            lambda _driver: avatar.get_attribute("data-state") == "listening"
        )
        listening_cue = self.browser.find_element(
            By.CSS_SELECTOR,
            ".voice-assistant__attitude-cue--listening",
        )
        self._check_equal(listening_cue.value_of_css_property("display"), "flex")
        query = self.browser.find_element(By.ID, "voice-assistant-query")
        self.wait.until(
            lambda _driver: query.get_attribute("value") == "classificatie MRI"
        )
        self._check_equal(
            self.browser.execute_script("return window.__recognitionLanguage;"),
            "nl-NL",
        )

    def test_thinking_attitude_is_visible_during_corpus_search(self) -> None:
        """Giovani visibly adopts the thinking attitude while RAG is running."""
        self._open_dashboard()
        self.browser.find_element(By.ID, "voice-assistant-query").send_keys(
            "classification MRI",
        )
        self.browser.find_element(By.ID, "voice-assistant-submit").click()

        avatar = self.browser.find_element(By.ID, "voice-assistant-avatar")
        self.wait.until(
            lambda _driver: avatar.get_attribute("data-state") == "thinking"
        )
        thinking_cue = self.browser.find_element(
            By.CSS_SELECTOR,
            ".voice-assistant__attitude-cue--thinking",
        )
        self._check_equal(thinking_cue.value_of_css_property("display"), "flex")
        self.wait.until(
            lambda driver: driver.find_elements(
                By.CSS_SELECTOR,
                ".voice-assistant__recommendation",
            ),
        )

    def test_text_to_speech_animates_giovani_lips(self) -> None:
        """Reading a result moves Giovani from speaking back to idle."""
        self._open_dashboard()
        query = self.browser.find_element(By.ID, "voice-assistant-query")
        query.send_keys("classification MRI")
        self.browser.find_element(By.ID, "voice-assistant-submit").click()
        read_button = self.wait.until(
            lambda driver: driver.find_element(By.ID, "voice-assistant-read"),
        )
        self.wait.until(lambda _driver: read_button.is_enabled())
        self.browser.execute_script(
            "arguments[0].scrollIntoView({block: 'center'}); arguments[0].click();",
            read_button,
        )

        avatar = self.browser.find_element(By.ID, "voice-assistant-avatar")
        self.wait.until(
            lambda _driver: avatar.get_attribute("data-state") == "speaking"
        )
        self.wait.until(
            lambda _driver: (
                avatar.get_attribute("data-viseme")
                in {"closed", "open", "wide", "round"}
            ),
        )
        self.wait.until(
            lambda driver: float(
                driver.execute_script(
                    "return parseFloat(getComputedStyle("
                    "document.querySelector('.voice-avatar__mouth-mask')).opacity);",
                ),
            )
            > MIN_MOUTH_OPACITY,
        )
        self.wait.until(
            lambda _driver: avatar.get_attribute("data-state") == "success"
        )
        success_cue = self.browser.find_element(
            By.CSS_SELECTOR,
            ".voice-assistant__attitude-cue--success",
        )
        self._check_equal(success_cue.value_of_css_property("display"), "flex")
        self.wait.until(lambda _driver: avatar.get_attribute("data-state") == "idle")

    def test_reading_speed_controls_voice_and_lip_timing(self) -> None:
        """The chosen rate is shared by speech synthesis and the lip rig."""
        self._open_dashboard()
        Select(
            self.browser.find_element(By.ID, "voice-assistant-rate")
        ).select_by_value("1.5")
        query = self.browser.find_element(By.ID, "voice-assistant-query")
        query.send_keys("classification MRI")
        self.browser.find_element(By.ID, "voice-assistant-submit").click()
        read_button = self.browser.find_element(By.ID, "voice-assistant-read")
        self.wait.until(lambda _driver: read_button.is_enabled())
        self.browser.execute_script(
            "arguments[0].scrollIntoView({block: 'center'}); arguments[0].click();",
            read_button,
        )

        avatar = self.browser.find_element(By.ID, "voice-assistant-avatar")
        self.wait.until(
            lambda _driver: avatar.get_attribute("data-state") == "speaking"
        )
        self.wait.until(
            lambda _driver: int(avatar.get_attribute("data-lip-index") or "0") > 0
        )
        self._check_equal(
            self.browser.execute_script("return window.__speechRate;"),
            1.5,
        )
        self._check_equal(avatar.get_attribute("data-reading-rate"), "1.5")
        self._check_equal(
            self.browser.execute_script(
                "return window.localStorage.getItem('aimer-assistant-rate');",
            ),
            "1.5",
        )

    def test_guided_demo_runs_dictation_analysis_and_spoken_response(self) -> None:
        """The demo chains TTS, STT, RAG rendering, and lip-synced playback."""
        self._open_dashboard()
        Select(
            self.browser.find_element(By.ID, "voice-assistant-language")
        ).select_by_value(
            "fr-FR",
        )
        demo_button = self.browser.find_element(By.ID, "voice-assistant-demo")
        demo_button.click()

        self.wait.until(
            lambda _driver: demo_button.get_attribute("aria-pressed") == "true"
        )
        self.wait.until(
            lambda driver: (
                driver.find_element(By.ID, "voice-assistant-query").get_attribute(
                    "value",
                )
                == "classificatie MRI"
            ),
        )
        self.wait.until(
            lambda driver: driver.find_elements(
                By.CSS_SELECTOR,
                ".voice-assistant__recommendation",
            ),
        )
        self.wait.until(
            lambda _driver: demo_button.get_attribute("aria-pressed") == "false"
        )

        steps = self.browser.find_elements(By.CSS_SELECTOR, "[data-demo-step]")
        self._check_equal(len(steps), 3)
        self._check(all("is-complete" in step.get_attribute("class") for step in steps))
        status = self.browser.find_element(By.ID, "voice-assistant-status")
        self._check("Démonstration terminée" in status.text)
