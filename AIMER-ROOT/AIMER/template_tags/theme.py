# Copyright (c) 2026 AIMER contributors.
"""Custom template tags and filters for theme utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from django import template
from django.contrib.auth.decorators import user_passes_test
from django.contrib.auth.models import AbstractBaseUser
from django.http import HttpRequest
from django.utils.html import format_html
from django.utils.safestring import SafeString

from AIMER.template_helpers.theme import TemplateHelper

register = template.Library()


def _safe(value: Any) -> SafeString:
    return format_html("{}", value)


@register.simple_tag
def get_theme_variables(scope: str) -> SafeString:
    """Return a theme variable as a safe rendered string."""
    return _safe(TemplateHelper.get_theme_variables(scope))


@register.simple_tag
def get_theme_config(scope: str) -> SafeString:
    """Return a theme config value as a safe rendered string."""
    return _safe(TemplateHelper.get_theme_config(scope))


@register.filter
def filter_by_url(submenu: Sequence[dict[str, Any]] | None, url: Any) -> bool:
    """Return whether a menu tree matches the current URL."""
    if not submenu:
        return False

    targets = {url.path, getattr(url.resolver_match, "url_name", "")}
    for subitem in submenu:
        if subitem.get("url") in targets:
            return True
        nested = subitem.get("submenu")
        if nested and filter_by_url(nested, url):
            return True
    return False


@register.filter
def has_group(user: AbstractBaseUser, group: str) -> bool:
    """Return whether the user belongs to a group."""
    return user.groups.filter(name=group).exists()


@register.filter
def has_permission(user: AbstractBaseUser, permission: str) -> bool:
    """Return whether the user has a permission."""
    return user.has_perm(permission)


@register.filter(name="is_admin")
def is_admin(user: AbstractBaseUser) -> bool:
    """Return whether the user belongs to admin group."""
    return has_group(user, "admin")


@register.filter(name="admin_required")
def admin_required(view_func: Callable[..., Any]) -> Callable[..., Any]:
    """Require admin group to access a view."""
    return user_passes_test(is_admin, login_url="login")(view_func)


@register.filter(name="is_client")
def is_client(user: AbstractBaseUser) -> bool:
    """Return whether the user belongs to client group."""
    return has_group(user, "client")


@register.filter(name="client_required")
def client_required(view_func: Callable[..., Any]) -> Callable[..., Any]:
    """Require client group to access a view."""
    return user_passes_test(is_client, login_url="login")(view_func)


@register.filter(name="is_superuser")
def is_superuser(user: AbstractBaseUser) -> bool:
    """Return whether user is superuser."""
    return bool(user.is_superuser)


@register.filter(name="superuser_required")
def superuser_required(view_func: Callable[..., Any]) -> Callable[..., Any]:
    """Require superuser to access a view."""
    return user_passes_test(is_superuser, login_url="login")(view_func)


@register.filter(name="is_staff")
def is_staff(user: AbstractBaseUser) -> bool:
    """Return whether user is staff."""
    return bool(user.is_staff)


@register.filter(name="staff_required")
def staff_required(view_func: Callable[..., Any]) -> Callable[..., Any]:
    """Require staff to access a view."""
    return user_passes_test(is_staff, login_url="login")(view_func)


@register.simple_tag
def current_url(request: HttpRequest) -> str:
    """Return full request URL."""
    return request.build_absolute_uri()
