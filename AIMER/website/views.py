from AIMER.template_helpers.theme import TemplateHelper
from django.views.generic import TemplateView

from AIMER import TemplateLayout

"""
This file is a view controller for multiple pages as a module.
Here you can override the page view layout.
Refer to front_pages/urls.py file for more pages.
"""


class FrontPagesView(TemplateView):
    # Predefined function
    def get_context_data(self, **kwargs):
        # A function to init the global layout. It is defined in AIMER/__init__.py file
        context = TemplateLayout.init(self, super().get_context_data(**kwargs))

        # Update the context
        context.update(
            {
                "layout": "front",
                "layout_path": TemplateHelper.set_layout("layout_front.html", context),
                "active_url": self.request.path,  # Get the current url path (active URL) from request
            }
        )

        # map_context according to updated context values
        TemplateHelper.map_context(context)

        return context
