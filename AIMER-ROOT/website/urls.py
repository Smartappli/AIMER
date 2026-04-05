# Copyright (c) 2026 AIMER contributors.

"""URL routes for website front pages."""

from django.urls import path

from .views import DashboardView, FrontPagesView, RagRecommendationView

urlpatterns = [
    path(
        "",
        FrontPagesView.as_view(template_name="landing_page.html"),
        name="index",
    ),
    path("dashboard/", DashboardView.as_view(), name="dashboard"),
    path("api/rag/recommend/", RagRecommendationView.as_view(), name="rag-recommend"),
]
