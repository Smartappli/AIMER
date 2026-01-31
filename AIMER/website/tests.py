from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group, Permission
from django.http import HttpRequest, HttpResponse
from django.test import RequestFactory, TestCase
from django.utils.safestring import SafeString

from AIMER import TemplateLayout
from AIMER.context_processors import environment, get_cookie, language_code, my_setting
from AIMER.language_middleware import DefaultLanguageMiddleware
from AIMER.template_helpers.theme import TemplateHelper
from AIMER.template_tags import theme as theme_tags
from templates.layout.bootstrap.layout_blank import TemplateBootstrapLayoutBlank
from templates.layout.bootstrap.layout_front import TemplateBootstrapLayoutFront
from templates.layout.bootstrap.layout_horizontal import (
    TemplateBootstrapLayoutHorizontal,
)
from templates.layout.bootstrap.layout_vertical import TemplateBootstrapLayoutVertical
from website.views import FrontPagesView


class TemplateHelperTests(TestCase):
    def test_init_context_populates_expected_keys(self) -> None:
        context: dict[str, object] = {}
        TemplateHelper.init_context(context)

        self.assertEqual(context["layout"], "vertical")
        self.assertEqual(context["navbar_type"], "fixed")
        self.assertIs(context["menu_fixed"], True)

    def test_map_context_for_horizontal_layout(self) -> None:
        context = {
            "layout": "horizontal",
            "header_type": "fixed",
            "navbar_type": "fixed",
            "menu_collapsed": False,
            "menu_fixed": False,
            "footer_fixed": True,
            "rtl_mode": False,
            "show_dropdown_onhover": True,
            "semiDark": True,
            "display_customizer": False,
            "content_layout": "wide",
            "navbar_detached": False,
        }

        TemplateHelper.map_context(context)

        self.assertEqual(context["header_type_class"], "layout-menu-fixed")
        self.assertEqual(context["navbar_type_class"], "")
        self.assertEqual(context["footer_fixed_class"], "layout-footer-fixed")
        self.assertEqual(context["container_class"], "container-fluid")
        self.assertEqual(context["navbar_detached_class"], "")

    def test_map_context_for_vertical_layout(self) -> None:
        context = {
            "layout": "vertical",
            "header_type": "static",
            "navbar_type": "hidden",
            "menu_collapsed": True,
            "menu_fixed": True,
            "footer_fixed": False,
            "rtl_mode": True,
            "show_dropdown_onhover": False,
            "semiDark": False,
            "display_customizer": True,
            "content_layout": "compact",
            "navbar_detached": True,
        }

        TemplateHelper.map_context(context)

        self.assertEqual(context["header_type_class"], "")
        self.assertEqual(context["navbar_type_class"], "layout-navbar-hidden")
        self.assertEqual(context["menu_collapsed_class"], "layout-menu-collapsed")
        self.assertEqual(context["menu_fixed_class"], "layout-menu-fixed")
        self.assertEqual(context["rtl_mode_value"], "rtl")
        self.assertEqual(context["display_customizer_class"], "")
        self.assertEqual(context["content_layout_class"], "layout-compact")
        self.assertEqual(context["navbar_detached_class"], "navbar-detached")

    def test_set_layout_uses_front_bootstrap(self) -> None:
        context: dict[str, object] = {}
        layout_path = TemplateHelper.set_layout("layout_front.html", context)

        self.assertEqual(layout_path, "layout/layout_front.html")
        self.assertEqual(context["layout"], "front")
        self.assertIs(context["is_front"], True)

    def test_template_bootstrap_variants(self) -> None:
        for bootstrap in (
            TemplateBootstrapLayoutBlank,
            TemplateBootstrapLayoutFront,
            TemplateBootstrapLayoutHorizontal,
            TemplateBootstrapLayoutVertical,
        ):
            context: dict[str, object] = {
                "navbar_detached": False,
                "menu_collapsed": False,
                "menu_fixed": False,
                "footer_fixed": False,
                "rtl_mode": False,
                "show_dropdown_onhover": False,
                "semiDark": False,
                "display_customizer": False,
                "content_layout": "compact",
                "header_type": "static",
                "navbar_type": "static",
            }
            bootstrap.init(context)

            self.assertIn("layout", context)


class TemplateLayoutTests(TestCase):
    def test_template_layout_respects_rtl_cookie(self) -> None:
        request = RequestFactory().get("/dashboard/")
        request.COOKIES["django_text_direction"] = "rtl"
        layout = TemplateLayout()
        layout.request = request

        context = layout.init({})

        self.assertEqual(context["rtl_mode"], True)
        self.assertEqual(context["layout_path"], "layout/layout_vertical.html")


class FrontPagesViewTests(TestCase):
    def test_front_pages_view_context(self) -> None:
        request = RequestFactory().get("/landing/")
        view = FrontPagesView()
        view.request = request

        context = view.get_context_data()

        self.assertEqual(context["layout"], "front")
        self.assertEqual(context["active_url"], "/landing/")
        self.assertEqual(context["layout_path"], "layout/layout_front.html")


class ContextProcessorTests(TestCase):
    def test_context_processors_values(self) -> None:
        request = HttpRequest()
        request.LANGUAGE_CODE = "fr"
        request.COOKIES = {"session": "abc"}

        self.assertIn("MY_SETTING", my_setting(request))
        self.assertEqual(language_code(request)["LANGUAGE_CODE"], "fr")
        self.assertEqual(get_cookie(request)["COOKIES"], {"session": "abc"})
        self.assertIn("ENVIRONMENT", environment(request))


class LanguageMiddlewareTests(TestCase):
    def test_middleware_sets_cookie_when_missing(self) -> None:
        request = RequestFactory().get("/")

        def get_response(request: HttpRequest) -> HttpResponse:
            return HttpResponse("ok")

        middleware = DefaultLanguageMiddleware(get_response)

        with patch("AIMER.language_middleware.activate") as activate_mock:
            response = middleware(request)

        activate_mock.assert_called_once()
        self.assertIn("django_language", response.cookies)

    def test_middleware_skips_when_cookie_present(self) -> None:
        request = RequestFactory().get("/")
        request.COOKIES["django_language"] = "en"

        def get_response(request: HttpRequest) -> HttpResponse:
            return HttpResponse("ok")

        middleware = DefaultLanguageMiddleware(get_response)

        with patch("AIMER.language_middleware.activate") as activate_mock:
            response = middleware(request)

        activate_mock.assert_not_called()
        self.assertNotIn("django_language", response.cookies)


class TemplateTagTests(TestCase):
    def setUp(self) -> None:
        self.factory = RequestFactory()
        self.user_model = get_user_model()

    def test_theme_tags_return_safe_values(self) -> None:
        theme_name = theme_tags.get_theme_variables("template_name")
        layout = theme_tags.get_theme_config("layout")

        self.assertIsInstance(theme_name, SafeString)
        self.assertEqual(str(theme_name), "AIMER")
        self.assertEqual(str(layout), "vertical")

    def test_filter_by_url_matches_nested_submenu(self) -> None:
        submenu = [
            {
                "url": "/dashboard/",
                "submenu": [
                    {"url": "/nested/", "submenu": []},
                ],
            },
        ]
        resolver = SimpleNamespace(url_name="nested")
        url = SimpleNamespace(path="/nested/", resolver_match=resolver)

        self.assertIs(theme_tags.filter_by_url(submenu, url), True)

    def test_group_and_permission_filters(self) -> None:
        user = self.user_model.objects.create_user(username="alice")
        admin_group = Group.objects.create(name="admin")
        client_group = Group.objects.create(name="client")
        user.groups.add(admin_group)

        permission = Permission.objects.filter(codename="add_user").first()
        if permission is not None:
            user.user_permissions.add(permission)

        self.assertIs(theme_tags.has_group(user, "admin"), True)
        self.assertIs(theme_tags.is_admin(user), True)
        self.assertIs(theme_tags.is_client(user), False)
        self.assertIs(theme_tags.has_permission(user, "auth.add_user"), True)

        user.groups.add(client_group)
        self.assertIs(theme_tags.is_client(user), True)

    def test_role_required_decorators_allow_access(self) -> None:
        def view(request: HttpRequest) -> HttpResponse:
            return HttpResponse("ok")

        admin_user = self.user_model.objects.create_user(username="admin")
        admin_user.groups.add(Group.objects.create(name="admin"))
        admin_request = self.factory.get("/")
        admin_request.user = admin_user

        client_user = self.user_model.objects.create_user(username="client")
        client_user.groups.add(Group.objects.create(name="client"))
        client_request = self.factory.get("/")
        client_request.user = client_user

        staff_user = self.user_model.objects.create_user(
            username="staff",
            is_staff=True,
        )
        staff_request = self.factory.get("/")
        staff_request.user = staff_user

        super_user = self.user_model.objects.create_user(
            username="super",
            is_superuser=True,
        )
        super_request = self.factory.get("/")
        super_request.user = super_user

        self.assertEqual(
            theme_tags.admin_required(view)(admin_request).status_code,
            200,
        )
        self.assertEqual(
            theme_tags.client_required(view)(client_request).status_code,
            200,
        )
        self.assertEqual(
            theme_tags.staff_required(view)(staff_request).status_code,
            200,
        )
        self.assertEqual(
            theme_tags.superuser_required(view)(super_request).status_code,
            200,
        )

    def test_user_flags_filters(self) -> None:
        user = self.user_model.objects.create_user(
            username="flags",
            is_superuser=True,
            is_staff=True,
        )

        self.assertIs(theme_tags.is_superuser(user), True)
        self.assertIs(theme_tags.is_staff(user), True)

    def test_current_url_tag(self) -> None:
        request = self.factory.get("/test/")
        self.assertEqual(theme_tags.current_url(request), "http://testserver/test/")
