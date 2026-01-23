from AIMER.template_helpers.theme import TemplateHelper
from django.views.generic import TemplateView

from AIMER import TemplateLayout

"""
This file is a view controller for multiple pages as a module.
Here you can override the page view layout.
Refer to auth/urls.py file for more pages.
"""


class AuthView(TemplateView):
    # Predefined function
    def get_context_data(self, **kwargs):
        # A function to init the global layout. It is defined in common_layers/__init__.py file
        context = TemplateLayout.init(self, super().get_context_data(**kwargs))

        # Update the context
        context.update(
            {
                "layout_path": TemplateHelper.set_layout(
                    "layout_blank.html", context,
                ),
            },
        )

        return context
