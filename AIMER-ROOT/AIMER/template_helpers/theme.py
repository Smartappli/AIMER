# Copyright (c) 2026 AIMER contributors.
"""Theme helper utilities used by templates."""

from __future__ import annotations

from importlib import import_module, util
from pathlib import Path
from typing import Any

from django.conf import settings


class TemplateHelper:
    """Utilities for building and mapping template context."""

    @staticmethod
    def init_context(context: dict[str, Any]) -> dict[str, Any]:
        """Populate a context dict with template settings."""
        context.update(
            {
                "layout": settings.TEMPLATE_CONFIG.get("layout"),
                "primary_color": settings.TEMPLATE_CONFIG.get("primary_color"),
                "theme": settings.TEMPLATE_CONFIG.get("theme"),
                "skins": settings.TEMPLATE_CONFIG.get("my_skins"),
                "semiDark": settings.TEMPLATE_CONFIG.get("has_semi_dark"),
                "rtl_mode": settings.TEMPLATE_CONFIG.get("rtl_mode"),
                "has_customizer": settings.TEMPLATE_CONFIG.get("has_customizer"),
                "display_customizer": settings.TEMPLATE_CONFIG.get(
                    "display_customizer",
                ),
                "content_layout": settings.TEMPLATE_CONFIG.get("content_layout"),
                "navbar_type": settings.TEMPLATE_CONFIG.get("navbar_type"),
                "header_type": settings.TEMPLATE_CONFIG.get("header_type"),
                "menu_fixed": settings.TEMPLATE_CONFIG.get("menu_fixed"),
                "menu_collapsed": settings.TEMPLATE_CONFIG.get("menu_collapsed"),
                "footer_fixed": settings.TEMPLATE_CONFIG.get("footer_fixed"),
                "show_dropdown_onhover": settings.TEMPLATE_CONFIG.get(
                    "show_dropdown_onhover",
                ),
                "customizer_controls": settings.TEMPLATE_CONFIG.get(
                    "customizer_controls",
                ),
            },
        )
        return context

    @staticmethod
    def map_context(context: dict[str, Any]) -> None:
        """Map context values to helper CSS classes and rendered booleans."""
        is_horizontal = context.get("layout") == "horizontal"
        is_vertical = context.get("layout") == "vertical"

        context["header_type_class"] = (
            "layout-menu-fixed"
            if is_horizontal and context.get("header_type") == "fixed"
            else ""
        )

        if is_horizontal:
            context["navbar_type_class"] = ""
        elif context.get("navbar_type") == "fixed":
            context["navbar_type_class"] = "layout-navbar-fixed"
        elif context.get("navbar_type") == "static":
            context["navbar_type_class"] = ""
        else:
            context["navbar_type_class"] = "layout-navbar-hidden"

        context["menu_collapsed_class"] = (
            "layout-menu-collapsed" if context.get("menu_collapsed") else ""
        )
        context["menu_fixed_class"] = (
            "layout-menu-fixed" if is_vertical and context.get("menu_fixed") else ""
        )
        context["footer_fixed_class"] = (
            "layout-footer-fixed" if context.get("footer_fixed") else ""
        )

        if context.get("rtl_mode"):
            context["rtl_mode_value"] = "rtl"
            context["text_direction_value"] = "rtl"
        else:
            context["rtl_mode_value"] = "ltr"
            context["text_direction_value"] = "ltr"

        context["show_dropdown_onhover_value"] = (
            "true" if context.get("show_dropdown_onhover") else "false"
        )
        context["semi_dark_value"] = "true" if context.get("semiDark") else "false"
        context["display_customizer_class"] = (
            "" if context.get("display_customizer") else "customizer-hide"
        )

        if context.get("content_layout") == "wide":
            context["container_class"] = "container-fluid"
            context["content_layout_class"] = "layout-wide"
        else:
            context["container_class"] = "container-xxl"
            context["content_layout_class"] = "layout-compact"

        context["navbar_detached_class"] = (
            "navbar-detached" if context.get("navbar_detached") else ""
        )

    @staticmethod
    def get_theme_variables(scope: str) -> Any:
        """Return an entry from THEME_VARIABLES."""
        return settings.THEME_VARIABLES[scope]

    @staticmethod
    def get_theme_config(scope: str) -> Any:
        """Return an entry from TEMPLATE_CONFIG."""
        return settings.TEMPLATE_CONFIG[scope]

    @staticmethod
    def set_layout(view: str, context: dict[str, Any] | None = None) -> str:
        """Resolve and initialize layout bootstrap class for a page view."""
        context = {} if context is None else context
        layout = Path(view).stem.split("/")[0]
        base_module = f"templates.{settings.THEME_LAYOUT_DIR.replace('/', '.')}"
        module = f"{base_module}.bootstrap.{layout}"

        if util.find_spec(module) is None:
            module = f"{base_module}.bootstrap.default"
            class_name = "TemplateBootstrapDefault"
        else:
            class_name = f"TemplateBootstrap{layout.title().replace('_', '')}"

        template_bootstrap = TemplateHelper.import_class(module, class_name)
        template_bootstrap.init(context)
        return f"{settings.THEME_LAYOUT_DIR}/{view}"

    @staticmethod
    def import_class(from_module: str, import_class_name: str) -> Any:
        """Import and return a class by module and class names."""
        module = import_module(from_module)
        return getattr(module, import_class_name)
