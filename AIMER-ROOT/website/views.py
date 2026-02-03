from AIMER.template_helpers.theme import TemplateHelper
from django.views.generic import TemplateView

from AIMER import TemplateLayout

"""
This file is a view controller for multiple pages as a module.
Here you can override the page view layout.
Refer to front_pages/urls.py file for more pages.
"""


class FrontPagesView(TemplateView):
    """Base view for AIMER "front" pages.

    Overrides `get_context_data()` to:
    1) Initialize the global layout context via `TemplateLayout.init()`.
    2) Add layout-related variables expected by the front templates.
    3) Map/normalize context values via `TemplateHelper.map_context()`.

    Typical usage is to set `template_name` in URL patterns or subclasses.
    """

    def get_context_data(self, **kwargs):
        """Build template context for front pages.

        Args:
            **kwargs: Context parameters passed by Django (e.g. URL kwargs).

        Returns:
            A context dictionary enriched with:
                - layout: "front"
                - layout_path: resolved layout template path
                - active_url: current request path
            and any global layout context initialized by `TemplateLayout.init()`.

        """
        context = TemplateLayout.init(self, super().get_context_data(**kwargs))

        # Update the context
        context.update(
            {
                "layout": "front",
                "layout_path": TemplateHelper.set_layout(
                    "layout_front.html",
                    context,
                ),
                "active_url": self.request.path,  # Get the current url path (active URL) from request
            },
        )

        # map_context according to updated context values
        TemplateHelper.map_context(context)

        return context
