# Copyright (c) 2026 AIMER contributors.
"""Vertical bootstrap layout context initializers."""

import json

from AIMER.template_helpers.theme import TemplateHelper
from django.conf import settings

menu_file_path = (
    settings.BASE_DIR
    / "templates"
    / "layout"
    / "partials"
    / "menu"
    / "vertical"
    / "json"
    / "vertical_menu.json"
)


class TemplateBootstrapLayoutVertical:
    """Provide setup helpers for the vertical bootstrap layout."""

    @staticmethod
    def init(context: dict[str, object]) -> dict[str, object]:
        """
        Initialize the template context for vertical layout.

        Returns:
            Updated context dictionary.

        """
        context.update(
            {
                "layout": "vertical",
                "content_navbar": True,
                "is_navbar": True,
                "is_menu": True,
                "is_footer": True,
                "navbar_detached": True,
            },
        )

        # map_context according to updated context values
        TemplateHelper.map_context(context)

        TemplateBootstrapLayoutVertical.init_menu_data(context)

        return context

    @staticmethod
    def init_menu_data(context: dict[str, object]) -> None:
        """Load vertical menu JSON data and inject it into context."""
        # Load the menu data from the JSON file
        menu_data = json.load(menu_file_path.open()) if menu_file_path.exists() else []

        # Updated context with menu_data
        context.update({"menu_data": menu_data})
