from AIMER.template_helpers.theme import TemplateHelper

"""
This is an entry and Bootstrap class for the theme level.
The init() function will be called in web_project/__init__.py
"""


class TemplateBootstrapLayoutFront:
    def init(context):
        context.update(
            {
                "layout": "front",
                "is_front": True,
                "display_customizer": False,
                "content_layout": "wide",
                "navbar_type": "fixed",
            }
        )
        # map_context according to updated context values
        TemplateHelper.map_context(context)

        return context
