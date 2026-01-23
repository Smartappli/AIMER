"""Authentication views base controller.

This module defines `AuthView`, a shared base `TemplateView` for authentication
pages (login, register, verify email, reset password, etc.).

It centralizes layout initialization and ensures auth pages use a minimal/blank
layout by setting:
- `layout_path` to the blank layout template (e.g. "layout_blank.html")

See `auth/urls.py` for the routes that use this view or its subclasses.
"""

from AIMER.template_helpers.theme import TemplateHelper
from django.views.generic import TemplateView

from AIMER import TemplateLayout


class AuthView(TemplateView):
    """Base view for authentication-related pages.

    Overrides `get_context_data()` to initialize the global layout context via
    `TemplateLayout.init()` and to enforce the blank layout for auth templates.
    """

    def get_context_data(self, **kwargs):
        """Build template context for auth pages.

        Args:
            **kwargs: Context parameters passed by Django (e.g. URL kwargs).

        Returns:
            A context dictionary enriched with:
                - layout_path: resolved blank layout template path
            and any global layout context initialized by `TemplateLayout.init()`.

        """
        context = TemplateLayout.init(self, super().get_context_data(**kwargs))

        # Update the context
        context.update(
            {"layout_path": TemplateHelper.set_layout("layout_blank.html", context)},
        )

        return context
