# Copyright (c) 2026 AIMER contributors.

"""URL configuration for framework project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/

Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))

"""

from django.contrib import admin
from django.db import DatabaseError, connection
from django.http import HttpRequest, JsonResponse
from django.urls import path
from django.views.decorators.http import require_safe


@require_safe
def healthz(_request: HttpRequest) -> JsonResponse:
    """Return a minimal liveness payload for deployment smoke tests."""
    return JsonResponse({"service": "FARM", "status": "ok"}, status=200)


@require_safe
def readyz(_request: HttpRequest) -> JsonResponse:
    """Return database-backed readiness for orchestrator probes."""
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
    except DatabaseError:
        return JsonResponse(
            {
                "service": "FARM",
                "status": "unavailable",
                "checks": {"database": "error"},
            },
            status=503,
        )
    return JsonResponse(
        {"service": "FARM", "status": "ok", "checks": {"database": "ok"}},
        status=200,
    )


urlpatterns = [
    path("healthz/", healthz, name="healthz"),
    path("readyz/", readyz, name="readyz"),
    path("admin/", admin.site.urls),
]
