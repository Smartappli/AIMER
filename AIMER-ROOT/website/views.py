# Copyright (c) 2026 AIMER contributors.

"""View controllers for website front pages."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from AIMER import TemplateLayout
from AIMER.template_helpers.theme import TemplateHelper
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views import View
from django.views.generic import TemplateView

from RAG.recommender import recommend_models_for_query
from RAG.timm_articles import ensure_timm_article_index_is_fresh, load_timm_article_index

PROJECT_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "AIMER-ROOT": ("vit", "transformer", "swin", "beit", "eva"),
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
    "AIMER-ROOT": "Dashboard produit et visualisation transverse des projets.",
    "MAGE": "Moteur ML/IA et catalogue de modèles de vision.",
    "FARM": "Workflows data/platform et stratégies d'optimisation d'entraînement.",
}


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
    """Return the RAG PDF directory used to feed project dashboards."""
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


class RagRecommendationView(View):
    """API endpoint returning ranked model recommendations from the RAG corpus."""

    def get(self, request: HttpRequest, *args: object, **kwargs: object) -> JsonResponse:
        """Return recommendation JSON for a natural-language clinician query."""
        del args, kwargs
        query = (request.GET.get("q") or "").strip()
        if not query:
            return JsonResponse(
                {"error": "Missing required query parameter: q"},
                status=400,
            )

        try:
            top_k = int(request.GET.get("top_k", "3"))
        except ValueError:
            top_k = 3
        top_k = max(1, min(top_k, 10))

        payload = recommend_models_for_query(query=query, top_k=top_k)
        return JsonResponse(payload.model_dump(), status=200)
