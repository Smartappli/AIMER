# Copyright (c) 2026 AIMER contributors.
"""Tests for the Django-to-RAG service client boundary."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
from django.test import SimpleTestCase, override_settings

from website.rag_client import (
    RagServiceUnavailableError,
    recommend_models,
    runtime_status,
)


class RagClientTests(SimpleTestCase):
    """Tests for remote RAG service calls from Django."""

    @override_settings(RAG_SERVICE_URL="http://rag-service:8000")
    @patch("website.rag_client.httpx.request")
    def test_recommend_models_calls_remote_service(self, mock_request) -> None:
        """Ensure recommendation requests use the remote RAG API contract."""
        mock_request.return_value = httpx.Response(
            200,
            json={
                "query": "classification mri",
                "recommended_models": [{"model_name": "ResNet"}],
            },
        )

        payload = recommend_models(
            query="classification mri",
            top_k=2,
            strict_openrag=True,
            language="nl",
        )

        self.assertEqual(payload["query"], "classification mri")
        self.assertEqual(mock_request.call_args.args[0], "POST")
        self.assertEqual(
            mock_request.call_args.args[1],
            "http://rag-service:8000/recommend",
        )
        self.assertEqual(
            json.loads(json.dumps(mock_request.call_args.kwargs["json"])),
            {
                "query": "classification mri",
                "top_k": 2,
                "strict_openrag": True,
                "language": "nl",
            },
        )

    @override_settings(RAG_SERVICE_URL="http://rag-service:8000")
    @patch("website.rag_client.httpx.request")
    def test_runtime_status_calls_remote_readiness(self, mock_request) -> None:
        """Ensure runtime health uses the service readiness endpoint."""
        mock_request.return_value = httpx.Response(
            200,
            json={"ready": True, "status": {"openrag_installed": True}},
        )

        payload = runtime_status()

        self.assertTrue(payload["ready"])
        self.assertEqual(mock_request.call_args.args[0], "GET")
        self.assertEqual(
            mock_request.call_args.args[1],
            "http://rag-service:8000/readyz",
        )

    @override_settings(RAG_SERVICE_URL="file:///tmp/rag.sock")
    @patch("website.rag_client.httpx.request")
    def test_rejects_non_http_remote_service_url(self, mock_request) -> None:
        """Ensure only HTTP(S) service URLs can cross the client boundary."""
        with self.assertRaisesRegex(RagServiceUnavailableError, "HTTP\\(S\\)"):
            runtime_status()

        mock_request.assert_not_called()

    @override_settings(RAG_SERVICE_URL="http://rag-service:8000")
    @patch("website.rag_client.settings.IS_PRODUCTION", True)
    @patch("website.rag_client.httpx.request")
    def test_production_rejects_http_remote_service_url(self, mock_request) -> None:
        """Production traffic must not fall back to plaintext RAG transport."""
        with self.assertRaisesRegex(RagServiceUnavailableError, "use HTTPS"):
            runtime_status()

        mock_request.assert_not_called()

    @override_settings(
        RAG_SERVICE_URL="https://rag-service:8000",
        RAG_SERVICE_CA_CERT_PATH="/etc/aimer-rag-ca/ca.crt",
    )
    @patch("website.rag_client.httpx.Client")
    def test_remote_rag_service_uses_configured_ca(self, mock_client) -> None:
        """The internal CA bundle must verify the encrypted RAG connection."""
        client = mock_client.return_value.__enter__.return_value
        client.request.return_value = httpx.Response(
            200,
            json={"ready": True, "status": {"openrag_installed": True}},
        )

        payload = runtime_status()

        self.assertTrue(payload["ready"])
        mock_client.assert_called_once_with(verify="/etc/aimer-rag-ca/ca.crt")
        client.request.assert_called_once()
