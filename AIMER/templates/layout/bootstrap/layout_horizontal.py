from django.conf import settings
import json


from AIMER.template_helpers.theme import TemplateHelper

menu_file_path = (
    settings.BASE_DIR
    / "templates"
    / "layout"
    / "partials"
    / "menu"
    / "horizontal"
    / "json"
    / "horizontal_menu.json"
)


"""
This is an entry and Bootstrap class for the theme level.
The init() function will be called in web_project/__init__.py
"""


class TemplateBootstrapLayoutHorizontal:
    def init(context):
        context.update(
            {
                "layout": "horizontal",
                "is_navbar": True,
                "navbar_full": True,
                "is_menu": True,
                "menu_horizontal": True,
                "is_footer": True,
                "navbar_detached": False,
            }
        )
        # map_context according to updated context values
        TemplateHelper.map_context(context)

        # Init menu data and update context
        TemplateBootstrapLayoutHorizontal.init_menu_data(context)

        return context

    def init_menu_data(context):
        # Load the menu data from the JSON file
        menu_data = (
            json.load(menu_file_path.open()) if menu_file_path.exists() else []
        )

        # Updated context with menu_data
        context.update({"menu_data": menu_data})
