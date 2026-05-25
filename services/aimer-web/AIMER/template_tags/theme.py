# Copyright (c) 2026 AIMER contributors.
"""Custom template tags and filters for theme utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from django import template
from django.contrib.auth.decorators import user_passes_test
from django.utils.html import format_html

from AIMER.template_helpers.theme import TemplateHelper

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from django.contrib.auth.models import AbstractBaseUser
    from django.http import HttpRequest
    from django.utils.safestring import SafeString

register = template.Library()


def _safe(value: object) -> SafeString:
    """
    Render a value as a Django safe string.

    Returns:
        Safely escaped string representation.

    """
    return format_html("{}", value)


@register.simple_tag
def get_theme_variables(scope: str) -> SafeString:
    """
    Return a theme variable as a safe rendered string.

    Returns:
        Safely rendered theme variable.

    """
    return _safe(TemplateHelper.get_theme_variables(scope))


@register.simple_tag
def get_theme_config(scope: str) -> SafeString:
    """
    Return a theme config value as a safe rendered string.

    Returns:
        Safely rendered theme configuration value.

    """
    return _safe(TemplateHelper.get_theme_config(scope))


@register.filter
def filter_by_url(submenu: Sequence[dict[str, object]] | None, url: object) -> bool:
    """
    Return whether a menu tree matches the current URL.

    Returns:
        ``True`` if the URL exists in the menu tree, otherwise ``False``.

    """
    if not submenu:
        return False

    url_path = getattr(url, "path", "")
    resolver_match = getattr(url, "resolver_match", None)
    url_name = getattr(resolver_match, "url_name", "")
    targets = {url_path, url_name}

    for subitem in submenu:
        if subitem.get("url") in targets:
            return True
        nested = subitem.get("submenu")
        if isinstance(nested, list) and filter_by_url(nested, url):
            return True
    return False


@register.filter
def has_group(user: AbstractBaseUser, group: str) -> bool:
    """
    Return whether the user belongs to a group.

    Returns:
        ``True`` when the user belongs to the named group.

    """
    return user.groups.filter(name=group).exists()


@register.filter
def has_permission(user: AbstractBaseUser, permission: str) -> bool:
    """
    Return whether the user has a permission.

    Returns:
        ``True`` when the user has the provided permission.

    """
    return user.has_perm(permission)


@register.filter(name="is_admin")
def is_admin(user: AbstractBaseUser) -> bool:
    """
    Return whether the user belongs to admin group.

    Returns:
        ``True`` when user is in the admin group.

    """
    return has_group(user, "admin")


@register.filter(name="admin_required")
def admin_required(view_func: Callable[..., object]) -> Callable[..., object]:
    """
    Require admin group to access a view.

    Returns:
        Wrapped view enforcing the admin-group check.

    """
    return user_passes_test(is_admin, login_url="login")(view_func)


@register.filter(name="is_client")
def is_client(user: AbstractBaseUser) -> bool:
    """
    Return whether the user belongs to client group.

    Returns:
        ``True`` when user is in the client group.

    """
    return has_group(user, "client")


@register.filter(name="client_required")
def client_required(view_func: Callable[..., object]) -> Callable[..., object]:
    """
    Require client group to access a view.

    Returns:
        Wrapped view enforcing the client-group check.

    """
    return user_passes_test(is_client, login_url="login")(view_func)


@register.filter(name="is_superuser")
def is_superuser(user: AbstractBaseUser) -> bool:
    """
    Return whether user is superuser.

    Returns:
        ``True`` if the user is a superuser.

    """
    return bool(user.is_superuser)


@register.filter(name="superuser_required")
def superuser_required(view_func: Callable[..., object]) -> Callable[..., object]:
    """
    Require superuser to access a view.

    Returns:
        Wrapped view enforcing the superuser check.

    """
    return user_passes_test(is_superuser, login_url="login")(view_func)


@register.filter(name="is_staff")
def is_staff(user: AbstractBaseUser) -> bool:
    """
    Return whether user is staff.

    Returns:
        ``True`` if the user is staff.

    """
    return bool(user.is_staff)


@register.filter(name="staff_required")
def staff_required(view_func: Callable[..., object]) -> Callable[..., object]:
    """
    Require staff to access a view.

    Returns:
        Wrapped view enforcing the staff check.

    """
    return user_passes_test(is_staff, login_url="login")(view_func)


@register.simple_tag
def current_url(request: HttpRequest) -> str:
    """
    Return full request URL.

    Returns:
        Absolute URL for the current request.

    """
    return request.build_absolute_uri()
