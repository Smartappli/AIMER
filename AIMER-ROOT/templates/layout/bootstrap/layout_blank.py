# Copyright (c) 2026 AIMER contributors.
"""Blank bootstrap layout context initializers."""

from AIMER.template_helpers.theme import TemplateHelper


class TemplateBootstrapLayoutBlank:
    """Provide setup helpers for the blank bootstrap layout."""

    @staticmethod
    def init(context: dict[str, object]) -> dict[str, object]:
        """Initialize the template context for blank layout.

        Returns:
            Updated context dictionary.

        """
        context.update(
            {
                "layout": "blank",
                "content_layout": "wide",
                "display_customizer": False,
            },
        )
        # map_context according to updated context values
        TemplateHelper.map_context(context)

        return context
