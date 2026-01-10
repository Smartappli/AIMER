from django.conf import settings
from pprint import pprint
import os
from importlib import import_module, util


# Core TemplateHelper class
class TemplateHelper:
    # Init the Template Context using TEMPLATE_CONFIG
    def init_context(context):
        context.update(
            {
                "layout": settings.TEMPLATE_CONFIG.get("layout"),
                "primary_color": settings.TEMPLATE_CONFIG.get("primary_color"),
                "theme": settings.TEMPLATE_CONFIG.get("theme"),
                "skins": settings.TEMPLATE_CONFIG.get("my_skins"),
                "semiDark": settings.TEMPLATE_CONFIG.get("has_semi_dark"),
                "rtl_mode": settings.TEMPLATE_CONFIG.get("rtl_mode"),
                "has_customizer": settings.TEMPLATE_CONFIG.get("has_customizer"),
                "display_customizer": settings.TEMPLATE_CONFIG.get( "display_customizer" ),
                "content_layout": settings.TEMPLATE_CONFIG.get("content_layout"),
                "navbar_type": settings.TEMPLATE_CONFIG.get("navbar_type"),
                "header_type": settings.TEMPLATE_CONFIG.get("header_type"),
                "menu_fixed": settings.TEMPLATE_CONFIG.get("menu_fixed"),
                "menu_collapsed": settings.TEMPLATE_CONFIG.get("menu_collapsed"),
                "footer_fixed": settings.TEMPLATE_CONFIG.get("footer_fixed"),
                "show_dropdown_onhover": settings.TEMPLATE_CONFIG.get( "show_dropdown_onhover" ),
                "customizer_controls": settings.TEMPLATE_CONFIG.get( "customizer_controls" ),
            }
        )
        return context

    # ? Map context variables to template class/value/variables names
    def map_context(context):
        #! Header Type (horizontal support only)
        if context.get("layout") == "horizontal":
            if context.get("header_type") == "fixed":
                context["header_type_class"] = "layout-menu-fixed"
            elif context.get("header_type") == "static":
                context["header_type_class"] = ""
            else:
                context["header_type_class"] = ""
        else:
            context["header_type_class"] = ""

        #! Navbar Type (vertical/front support only)
        if context.get("layout") != "horizontal":
            if context.get("navbar_type") == "fixed":
                context["navbar_type_class"] = "layout-navbar-fixed"
            elif context.get("navbar_type") == "static":
                context["navbar_type_class"] = ""
            else:
                context["navbar_type_class"] = "layout-navbar-hidden"
        else:
            context["navbar_type_class"] = ""

        # Menu collapsed
        context["menu_collapsed_class"] = (
            "layout-menu-collapsed" if context.get("menu_collapsed") else ""
        )

        #! Menu Fixed (vertical support only)
        if context.get("layout") == "vertical":
            if context.get("menu_fixed") is True:
                context["menu_fixed_class"] = "layout-menu-fixed"
            else:
                context["menu_fixed_class"] = ""

        # Footer Fixed
        context["footer_fixed_class"] = (
            "layout-footer-fixed" if context.get("footer_fixed") else ""
        )

        # RTL Mode/Layout
        context["rtl_mode_value"], context["text_direction_value"] = (
            ("rtl", "rtl") if context.get("rtl_mode") else ("ltr", "ltr")
        )

        #!  Show dropdown on hover (Horizontal menu)
        context["show_dropdown_onhover_value"] = (
            "true" if context.get("show_dropdown_onhover") else "false"
        )

        context["semi_dark_value"] = (
            "true" if context.get("semiDark") else "false"
        )

        # Display Customizer
        context["display_customizer_class"] = (
            "" if context.get("display_customizer") else "customizer-hide"
        )

        # Content Layout
        if context.get("content_layout") == "wide":
            context["container_class"] = "container-fluid"
            context["content_layout_class"] = "layout-wide"
        else:
            context["container_class"] = "container-xxl"
            context["content_layout_class"] = "layout-compact"

        # Detached Navbar
        if context.get("navbar_detached") == True:
            context["navbar_detached_class"] = "navbar-detached"
        else:
            context["navbar_detached_class"] = ""

    # Get theme variables by scope
    def get_theme_variables(scope):
        return settings.THEME_VARIABLES[scope]

    # Get theme config by scope
    def get_theme_config(scope):
        return settings.TEMPLATE_CONFIG[scope]

    # Set the current page layout and init the layout bootstrap file
    def set_layout(view, context={}):
        # Extract layout from the view path
        layout = os.path.splitext(view)[0].split("/")[0]

        # Get module path
        module = f"templates.{settings.THEME_LAYOUT_DIR.replace('/', '.')}.bootstrap.{layout}"

        # Check if the bootstrap file is exist
        if util.find_spec(module) is not None:
            # Auto import and init the default bootstrap.py file from the theme
            TemplateBootstrap = TemplateHelper.import_class(
                module, f"TemplateBootstrap{layout.title().replace('_', '')}"
            )
            TemplateBootstrap.init(context)
        else:
            module = f"templates.{settings.THEME_LAYOUT_DIR.replace('/', '.')}.bootstrap.default"

            TemplateBootstrap = TemplateHelper.import_class(
                module, "TemplateBootstrapDefault"
            )
            TemplateBootstrap.init(context)

        return f"{settings.THEME_LAYOUT_DIR}/{view}"

    # Import a module by string
    def import_class(fromModule, import_className):
        pprint(f"Loading {import_className} from {fromModule}")
        module = import_module(fromModule)
        return getattr(module, import_className)
