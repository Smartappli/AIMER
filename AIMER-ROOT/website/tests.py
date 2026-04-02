# Copyright (c) 2026 AIMER contributors.

"""Tests for template helpers, context processors, middleware, and template tags."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from AIMER import TemplateLayout
from AIMER.context_processors import environment, get_cookie, language_code, my_setting
from AIMER.language_middleware import DefaultLanguageMiddleware
from AIMER.template_helpers.theme import TemplateHelper
from AIMER.template_tags import theme as theme_tags
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AbstractBaseUser, Group, Permission
from django.contrib.contenttypes.models import ContentType
from django.http import HttpRequest, HttpResponse
from django.test import RequestFactory, TestCase
from django.utils.safestring import SafeString
from templates.layout.bootstrap.layout_blank import TemplateBootstrapLayoutBlank
from templates.layout.bootstrap.layout_front import TemplateBootstrapLayoutFront
from templates.layout.bootstrap.layout_horizontal import (
    TemplateBootstrapLayoutHorizontal,
)
from templates.layout.bootstrap.layout_vertical import TemplateBootstrapLayoutVertical

from website.views import FrontPagesView


class BaseTestCase(TestCase):
    """Base test case with helper checks compatible with project Ruff rules."""

    def _check(self, condition: object, message: str = "Expectation failed") -> None:
        if not bool(condition):
            self.fail(message)

    def _check_equal(
        self,
        left: object,
        right: object,
        message: str = "Values are not equal",
    ) -> None:
        if left != right:
            self.fail(f"{message}: {left!r} != {right!r}")


class TemplateHelperTests(BaseTestCase):
    """Tests for TemplateHelper context initialization and mapping."""

    def test_init_context_populates_expected_keys(self) -> None:
        """Ensure init_context sets core default keys."""
        context: dict[str, object] = {}
        TemplateHelper.init_context(context)

        self._check_equal(context["layout"], "vertical")
        self._check_equal(context["navbar_type"], "fixed")
        self._check(context["menu_fixed"])

    def test_map_context_for_horizontal_layout(self) -> None:
        """Ensure horizontal layout values are mapped correctly."""
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

        self._check_equal(context["header_type_class"], "layout-menu-fixed")
        self._check(not (context["navbar_type_class"]))
        self._check_equal(context["footer_fixed_class"], "layout-footer-fixed")
        self._check_equal(context["container_class"], "container-fluid")
        self._check(not (context["navbar_detached_class"]))

    def test_map_context_for_vertical_layout(self) -> None:
        """Ensure vertical layout values are mapped correctly."""
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

        self._check(not (context["header_type_class"]))
        self._check_equal(context["navbar_type_class"], "layout-navbar-hidden")
        self._check_equal(context["menu_collapsed_class"], "layout-menu-collapsed")
        self._check_equal(context["menu_fixed_class"], "layout-menu-fixed")
        self._check_equal(context["rtl_mode_value"], "rtl")
        self._check(not (context["display_customizer_class"]))
        self._check_equal(context["content_layout_class"], "layout-compact")
        self._check_equal(context["navbar_detached_class"], "navbar-detached")

    def test_set_layout_uses_front_bootstrap(self) -> None:
        """Ensure set_layout resolves front layout metadata."""
        context: dict[str, object] = {}
        layout_path = TemplateHelper.set_layout("layout_front.html", context)

        self._check_equal(layout_path, "layout/layout_front.html")
        self._check_equal(context["layout"], "front")
        self._check(context["is_front"])

    def test_template_bootstrap_variants(self) -> None:
        """Ensure each bootstrap variant initializes a layout key."""
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

            self._check("layout" in context)


class TemplateLayoutTests(BaseTestCase):
    """Tests for TemplateLayout behavior."""

    def test_template_layout_respects_rtl_cookie(self) -> None:
        """Ensure RTL cookie is reflected in resulting context."""
        request = RequestFactory().get("/dashboard/")
        request.COOKIES["django_text_direction"] = "rtl"

        layout = TemplateLayout()
        layout.request = request

        context = layout.init({})

        self._check(context["rtl_mode"])
        self._check_equal(context["layout_path"], "layout/layout_vertical.html")


class FrontPagesViewTests(BaseTestCase):
    """Tests for front pages view context generation."""

    def test_front_pages_view_context(self) -> None:
        """Ensure front page view provides expected context values."""
        request = RequestFactory().get("/landing/")

        view = FrontPagesView()
        view.request = request
        view.args = ()
        view.kwargs = {}

        context = view.get_context_data()

        self._check_equal(context["layout"], "front")
        self._check_equal(context["active_url"], "/landing/")
        self._check_equal(context["layout_path"], "layout/layout_front.html")


class ContextProcessorTests(BaseTestCase):
    """Tests for project context processors."""

    def test_context_processors_values(self) -> None:
        """Ensure context processors return expected keys and values."""
        request = HttpRequest()
        request.LANGUAGE_CODE = "fr"
        request.COOKIES = {"session": "abc"}

        self._check("MY_SETTING" in my_setting(request))
        self._check_equal(language_code(request)["LANGUAGE_CODE"], "fr")
        self._check_equal(get_cookie(request)["COOKIES"], {"session": "abc"})
        self._check("ENVIRONMENT" in environment(request))


class LanguageMiddlewareTests(BaseTestCase):
    """Tests for default language middleware behavior."""

    def test_middleware_sets_cookie_when_missing(self) -> None:
        """Ensure middleware sets language cookie when absent."""
        request = RequestFactory().get("/")

        def get_response(_req: HttpRequest) -> HttpResponse:
            return HttpResponse("ok")

        middleware = DefaultLanguageMiddleware(get_response)

        with patch("AIMER.language_middleware.activate") as activate_mock:
            response = middleware(request)

        activate_mock.assert_called_once()
        self._check("django_language" in response.cookies)

    def test_middleware_skips_when_cookie_present(self) -> None:
        """Ensure middleware does not reset language when cookie exists."""
        request = RequestFactory().get("/")
        request.COOKIES["django_language"] = "en"

        def get_response(_req: HttpRequest) -> HttpResponse:
            return HttpResponse("ok")

        middleware = DefaultLanguageMiddleware(get_response)

        with patch("AIMER.language_middleware.activate") as activate_mock:
            response = middleware(request)

        activate_mock.assert_not_called()
        self._check("django_language" not in response.cookies)


class TemplateTagTests(BaseTestCase):
    """Tests for custom template tags and permission helpers."""

    def setUp(self) -> None:
        """Prepare commonly used factory and user model handles."""
        self.factory = RequestFactory()
        self.user_model = get_user_model()

    def _mkuser(self, username: str, **extra_fields: object) -> AbstractBaseUser:
        """Create a user with unique username/email for each test.

        Returns:
            Newly created user instance.

        """
        return self.user_model.objects.create_user(
            username=username,
            email=f"{username}@example.com",
            password=uuid4().hex,
            **extra_fields,
        )

    def test_theme_tags_return_safe_values(self) -> None:
        """Ensure theme tag helpers return expected safe string values."""
        theme_name = theme_tags.get_theme_variables("template_name")
        layout = theme_tags.get_theme_config("layout")

        self._check(isinstance(theme_name, SafeString))
        self._check_equal(str(theme_name), "AIMER")
        self._check_equal(str(layout), "vertical")

    def test_filter_by_url_matches_nested_submenu(self) -> None:
        """Ensure submenu URL matcher works on nested entries."""
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

        self._check(theme_tags.filter_by_url(submenu, url))

    def test_group_and_permission_filters(self) -> None:
        """Ensure group and permission helper filters evaluate properly."""
        user = self._mkuser("alice")

        admin_group = Group.objects.create(name="admin")
        client_group = Group.objects.create(name="client")
        user.groups.add(admin_group)

        ct = ContentType.objects.get_for_model(self.user_model)
        add_codename = f"add_{ct.model}"
        permission = Permission.objects.get(
            content_type=ct,
            codename=add_codename,
        )

        user.user_permissions.add(permission)
        user.refresh_from_db()

        self._check(theme_tags.has_group(user, "admin"))
        self._check(theme_tags.is_admin(user))
        self._check(not (theme_tags.is_client(user)))

        perm_label = f"{ct.app_label}.{permission.codename}"
        self._check(theme_tags.has_permission(user, perm_label))

        user.groups.add(client_group)
        self._check(theme_tags.is_client(user))

    def test_role_required_decorators_allow_access(self) -> None:
        """Ensure role decorators allow access for matching users."""

        def view(_request: HttpRequest) -> HttpResponse:
            return HttpResponse("ok")

        admin_user = self._mkuser("admin")
        admin_user.groups.add(Group.objects.create(name="admin"))
        admin_request = self.factory.get("/")
        admin_request.user = admin_user

        client_user = self._mkuser("client")
        client_user.groups.add(Group.objects.create(name="client"))
        client_request = self.factory.get("/")
        client_request.user = client_user

        staff_user = self._mkuser("staff", is_staff=True)
        staff_request = self.factory.get("/")
        staff_request.user = staff_user

        super_user = self._mkuser("super", is_superuser=True)
        super_request = self.factory.get("/")
        super_request.user = super_user

        self._check_equal(
            theme_tags.admin_required(view)(admin_request).status_code,
            HttpResponse().status_code,
        )
        self._check_equal(
            theme_tags.client_required(view)(client_request).status_code,
            HttpResponse().status_code,
        )
        self._check_equal(
            theme_tags.staff_required(view)(staff_request).status_code,
            HttpResponse().status_code,
        )
        self._check_equal(
            theme_tags.superuser_required(view)(super_request).status_code,
            HttpResponse().status_code,
        )

    def test_user_flags_filters(self) -> None:
        """Ensure superuser and staff filters mirror user flags."""
        user = self._mkuser("flags", is_superuser=True, is_staff=True)

        self._check(theme_tags.is_superuser(user))
        self._check(theme_tags.is_staff(user))

    def test_current_url_tag(self) -> None:
        """Ensure current_url returns the absolute URL for request."""
        request = self.factory.get("/test/")
        self._check_equal(theme_tags.current_url(request), request.build_absolute_uri())
