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

"""
This is an entry and Bootstrap class for the theme level.
The init() function will be called in web_project/__init__.py
"""


class TemplateBootstrapLayoutVertical:
    def init(context):
        context.update(
            {
                "layout": "vertical",
                "content_navbar": True,
                "is_navbar": True,
                "is_menu": True,
                "is_footer": True,
                "navbar_detached": True,
            }
        )

        # map_context according to updated context values
        TemplateHelper.map_context(context)

        TemplateBootstrapLayoutVertical.init_menu_data(context)

        return context

    def init_menu_data(context):
        # Load the menu data from the JSON file
        menu_data = json.load(menu_file_path.open()) if menu_file_path.exists() else []

        # Updated context with menu_data
        context.update({"menu_data": menu_data})
