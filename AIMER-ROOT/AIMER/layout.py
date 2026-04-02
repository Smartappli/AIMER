# Copyright (c) 2026 AIMER contributors.
"""Layout bootstrap helpers for template rendering."""

from __future__ import annotations

from typing import Any

from django.conf import settings

from AIMER.template_helpers.theme import TemplateHelper


class TemplateLayout:
    """Prepare a page context with layout-related values."""

    request: Any

    def init(self, context: dict[str, Any]) -> dict[str, Any]:
        """Initialize template context for the current request.

        Returns:
            Updated template context with resolved layout metadata.

        """
        context = TemplateHelper.init_context(context)
        layout = context["layout"]
        context.update(
            {
                "layout_path": TemplateHelper.set_layout(
                    f"layout_{layout}.html", context
                ),
                "rtl_mode": (
                    self.request.COOKIES.get("django_text_direction") == "rtl"
                    or settings.TEMPLATE_CONFIG.get("rtl_mode")
                ),
            },
        )
        TemplateHelper.map_context(context)
        return context
