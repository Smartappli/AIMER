"""View controllers for website front pages."""

from typing import Any

from AIMER import TemplateLayout
from AIMER.template_helpers.theme import TemplateHelper
from django.views.generic import TemplateView


class FrontPagesView(TemplateView):
    """Base view for AIMER front pages."""

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        """Build template context for front pages.

        Returns:
            Context dictionary enriched for front templates.
        """
        context = TemplateLayout.init(self, super().get_context_data(**kwargs))

        context.update(
            {
                "layout": "front",
                "layout_path": TemplateHelper.set_layout(
                    "layout_front.html",
                    context,
                ),
                "active_url": self.request.path,
            },
        )

        TemplateHelper.map_context(context)

        return context
