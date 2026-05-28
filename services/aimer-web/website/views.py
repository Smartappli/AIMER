# Copyright (c) 2026 AIMER contributors.

"""View controllers for website front pages."""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import time
from typing import Final, override

from AIMER import TemplateLayout
from AIMER.template_helpers.theme import TemplateHelper
from auth.security import audit_event
from django.conf import settings
from django.core.cache import cache
from django.db import DatabaseError, connection
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views import View
from django.views.generic import TemplateView
from RAG.timm_articles import (
    ensure_timm_article_index_is_fresh,
    load_timm_article_index,
)
from website.rag_client import (
    RagServiceUnavailableError,
    recommend_models,
    runtime_status,
)

PROJECT_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "aimer-web": ("vit", "transformer", "swin", "beit", "eva"),
    "MAGE": (
        "resnet",
        "convnext",
        "mobilenet",
        "efficientnet",
        "vit",
        "transformer",
    ),
    "FARM": (
        "adam",
        "lamb",
        "lars",
        "novograd",
        "madgrad",
        "lookahead",
        "optimizer",
    ),
}

PROJECT_DESCRIPTIONS: Final[dict[str, str]] = {
    "aimer-web": "Dashboard produit et visualisation transverse des services.",
    "MAGE": "Moteur ML/IA et catalogue de modèles de vision.",
    "FARM": "Workflows data/platform et stratégies d'optimisation d'entraînement.",
}


RAG_RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60


def _rag_rate_limit_exceeded(request: HttpRequest) -> bool:
    """Return whether the authenticated caller exceeded the RAG API limit."""
    limit = int(getattr(settings, "RAG_RECOMMENDATION_RATE_LIMIT_PER_MINUTE", 30))
    if limit <= 0:
        return False

    window = int(time() // RAG_RATE_LIMIT_WINDOW_SECONDS)
    key = f"rag-recommend:{request.user.pk}:{window}"
    if cache.add(key, 1, timeout=RAG_RATE_LIMIT_WINDOW_SECONDS):
        return False

    try:
        count = cache.incr(key)
    except ValueError:
        cache.set(key, 1, timeout=RAG_RATE_LIMIT_WINDOW_SECONDS)
        return False
    return count > limit


def _query_hash(query: str) -> str:
    """Return a stable non-reversible identifier for a user-provided query."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


class FrontPagesView(TemplateView):
    """Base view for AIMER front pages."""

    def get_context_data(self, **kwargs: object) -> dict[str, object]:
        """
        Build template context for front pages.

        Returns:
            Context dictionary enriched for front templates.

        """
        context = TemplateLayout.init(self, super().get_context_data(**kwargs))

        context.update(
            {
                "layout": "front",
                "layout_path": TemplateHelper.set_layout(
                    "layout_front.html",
                    context,
                ),
                "active_url": self.request.path,
            },
        )

        TemplateHelper.map_context(context)

        return context


def _rag_pdf_directory() -> Path:
    """
    Return the RAG PDF directory used to feed project dashboards.

    Returns:
        Path: Absolute path to the directory containing RAG PDF sources.

    """
    return Path(__file__).resolve().parent.parent / "RAG" / "data" / "pdfs"


def _discover_scientific_articles(pdf_dir: Path | None = None) -> list[str]:
    """
    Discover scientific article filenames from the RAG corpus.

    Args:
        pdf_dir: Optional override for testing.

    Returns:
        Sorted list of PDF filenames.

    """
    articles_dir = pdf_dir or _rag_pdf_directory()
    if pdf_dir is None:
        ensure_timm_article_index_is_fresh(pdf_directory=articles_dir)
    local_articles = (
        sorted(path.name for path in articles_dir.glob("*.pdf"))
        if articles_dir.exists()
        else []
    )
    timm_seed_articles = [
        f"{article['model_name']} - {article['paper_title']} ({article['paper_url']})"
        for article in load_timm_article_index()
    ]
    return sorted({*local_articles, *timm_seed_articles})


def _build_project_rag_index(
    projects: dict[str, tuple[str, ...]],
    articles: list[str],
) -> dict[str, list[str]]:
    """
    Build a mapping of project names to related scientific article filenames.

    Args:
        projects: Project keyword map.
        articles: Available article filenames.

    Returns:
        Project-indexed list of matched article filenames.

    """
    index: dict[str, list[str]] = {}
    normalized_articles = [(name, name.lower()) for name in articles]
    for project_name, keywords in projects.items():
        matches = [
            name
            for name, lowered in normalized_articles
            if any(keyword in lowered for keyword in keywords)
        ]
        index[project_name] = matches
    return index


class DashboardView(FrontPagesView):
    """Dashboard view: project-centric RAG of scientific articles."""

    template_name = "dashboard.html"

    def dispatch(
        self,
        request: HttpRequest,
        *args: object,
        **kwargs: object,
    ) -> HttpResponse:
        """
        Allow dashboard access only for authenticated users.

        Unauthenticated users are served the public landing page.

        Returns:
            HttpResponse: The dashboard response for authenticated users, or the
                landing page for unauthenticated users.

        """
        if not request.user.is_authenticated:
            return FrontPagesView.as_view(template_name="landing_page.html")(
                request,
                *args,
                **kwargs,
            )
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs: object) -> dict[str, object]:
        """
        Build dashboard context with project-to-article RAG mapping.

        Returns:
            Context including article index, counts, and project cards.

        """
        context = super().get_context_data(**kwargs)
        articles = _discover_scientific_articles()
        project_index = _build_project_rag_index(PROJECT_KEYWORDS, articles)
        context["total_scientific_articles"] = len(articles)
        context["project_cards"] = [
            {
                "name": project_name,
                "description": PROJECT_DESCRIPTIONS[project_name],
                "articles": project_index[project_name],
                "article_count": len(project_index[project_name]),
            }
            for project_name in PROJECT_KEYWORDS
        ]
        return context


class HealthCheckView(View):
    """Public liveness endpoint for deployment smoke tests."""

    @override
    def get(
        self,
        request: HttpRequest,
        *args: object,
        **kwargs: object,
    ) -> JsonResponse:
        del request, args, kwargs
        return JsonResponse({"service": "aimer-web", "status": "ok"}, status=200)


class ReadinessCheckView(View):
    """Public readiness endpoint for orchestrator traffic decisions."""

    @override
    def get(
        self,
        request: HttpRequest,
        *args: object,
        **kwargs: object,
    ) -> JsonResponse:
        del request, args, kwargs
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
        except DatabaseError:
            return JsonResponse(
                {
                    "service": "aimer-web",
                    "status": "unavailable",
                    "checks": {"database": "error"},
                },
                status=503,
            )
        return JsonResponse(
            {
                "service": "aimer-web",
                "status": "ok",
                "checks": {"database": "ok"},
            },
            status=200,
        )


class RagRecommendationView(View):
    """API endpoint returning ranked model recommendations from the RAG corpus."""

    @override
    def get(
        self,
        request: HttpRequest,
        *args: object,
        **kwargs: object,
    ) -> JsonResponse:
        """
        Return recommendation JSON for a natural-language clinician query.

        Returns:
            JsonResponse: A JSON payload containing ranked recommendations, or
                an error response when the query parameter is missing.

        """
        del args, kwargs
        if not request.user.is_authenticated:
            audit_event(
                "rag.recommendation.unauthenticated",
                request=request,
                metadata={"reason": "anonymous"},
            )
            return JsonResponse({"error": "Authentication required"}, status=401)
        if _rag_rate_limit_exceeded(request):
            audit_event(
                "rag.recommendation.rate_limited",
                request=request,
                user=request.user,
            )
            return JsonResponse({"error": "Rate limit exceeded"}, status=429)

        query = (request.GET.get("q") or "").strip()
        if not query:
            audit_event(
                "rag.recommendation.rejected",
                request=request,
                user=request.user,
                metadata={"reason": "missing_query"},
            )
            return JsonResponse(
                {"error": "Missing required query parameter: q"},
                status=400,
            )

        try:
            top_k = int(request.GET.get("top_k", "3"))
        except ValueError:
            top_k = 3
        top_k = max(1, min(top_k, 10))

        try:
            payload = recommend_models(
                query=query,
                top_k=top_k,
                strict_openrag=True,
            )
        except RagServiceUnavailableError as exc:
            audit_event(
                "rag.recommendation.unavailable",
                request=request,
                user=request.user,
                metadata={
                    "query_hash": _query_hash(query),
                    "top_k": top_k,
                    "reason": str(exc)[:256],
                },
            )
            return JsonResponse({"error": str(exc)}, status=503)
        audit_event(
            "rag.recommendation.requested",
            request=request,
            user=request.user,
            metadata={
                "query_hash": _query_hash(query),
                "top_k": top_k,
                "recommendation_count": len(payload.get("recommended_models") or []),
            },
        )
        return JsonResponse(payload, status=200)


class RagRuntimeHealthView(View):
    """API endpoint exposing OpenRAG runtime readiness."""

    @override
    def get(
        self,
        request: HttpRequest,
        *args: object,
        **kwargs: object,
    ) -> JsonResponse:
        del args, kwargs
        if not request.user.is_authenticated:
            audit_event(
                "rag.health.unauthenticated",
                request=request,
                metadata={"reason": "anonymous"},
            )
            return JsonResponse({"error": "Authentication required"}, status=401)
        if not request.user.is_staff:
            audit_event(
                "rag.health.forbidden",
                request=request,
                user=request.user,
                metadata={"reason": "non_staff"},
            )
            return JsonResponse({"error": "Staff access required"}, status=403)

        payload = runtime_status()
        audit_event(
            "rag.health.read",
            request=request,
            user=request.user,
            metadata={"ready": bool(payload.get("ready"))},
        )
        return JsonResponse(payload, status=200)
