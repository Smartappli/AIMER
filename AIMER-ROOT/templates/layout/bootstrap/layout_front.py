# Copyright (c) 2026 AIMER contributors.
"""Front bootstrap layout context initializers."""

from AIMER.template_helpers.theme import TemplateHelper


class TemplateBootstrapLayoutFront:
    """Provide setup helpers for the front bootstrap layout."""

    @staticmethod
    def init(context: dict[str, object]) -> dict[str, object]:
        """
        Initialize the template context for front layout.

        Returns:
            Updated context dictionary.

        """
        context.update(
            {
                "layout": "front",
                "is_front": True,
                "display_customizer": False,
                "content_layout": "wide",
                "navbar_type": "fixed",
            },
        )
        # map_context according to updated context values
        TemplateHelper.map_context(context)

        return context
