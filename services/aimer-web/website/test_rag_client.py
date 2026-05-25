# Copyright (c) 2026 AIMER contributors.
"""Tests for the Django-to-RAG service client boundary."""

from __future__ import annotations

import json
from types import TracebackType
from typing import Self
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from website.rag_client import recommend_models, runtime_status


class _JsonResponse:
    """Minimal context-manager response used to fake urllib calls."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc, traceback
        return False

    def read(self) -> bytes:
        """Return the fake response body."""
        return json.dumps(self.payload).encode("utf-8")


class RagClientTests(SimpleTestCase):
    """Tests for remote RAG service calls from Django."""

    @override_settings(RAG_SERVICE_URL="http://rag-service:8000")
    @patch("website.rag_client.urlopen")
    def test_recommend_models_calls_remote_service(self, mock_urlopen) -> None:
        """Ensure recommendation requests use the remote RAG API contract."""
        mock_urlopen.return_value = _JsonResponse(
            {
                "query": "classification mri",
                "recommended_models": [{"model_name": "ResNet"}],
            },
        )

        payload = recommend_models(
            query="classification mri",
            top_k=2,
            strict_openrag=True,
        )

        self.assertEqual(payload["query"], "classification mri")
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://rag-service:8000/recommend")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            {
                "query": "classification mri",
                "top_k": 2,
                "strict_openrag": True,
            },
        )

    @override_settings(RAG_SERVICE_URL="http://rag-service:8000")
    @patch("website.rag_client.urlopen")
    def test_runtime_status_calls_remote_readiness(self, mock_urlopen) -> None:
        """Ensure runtime health uses the service readiness endpoint."""
        mock_urlopen.return_value = _JsonResponse(
            {"ready": True, "status": {"openrag_installed": True}},
        )

        payload = runtime_status()

        self.assertTrue(payload["ready"])
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://rag-service:8000/readyz")
        self.assertEqual(request.get_method(), "GET")
