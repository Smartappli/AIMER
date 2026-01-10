from django.utils.safestring import mark_safe
from django import template
from AIMER.template_helpers.theme import TemplateHelper
from django.contrib.auth.decorators import user_passes_test

register = template.Library()


# Register tags as an adapter for the Theme class usage in the HTML template


@register.simple_tag
def get_theme_variables(scope):
    return mark_safe(TemplateHelper.get_theme_variables(scope))


@register.simple_tag
def get_theme_config(scope):
    return mark_safe(TemplateHelper.get_theme_config(scope))


@register.filter
def filter_by_url(submenu, url):
    if submenu:
        for subitem in submenu:
            subitem_url = subitem.get("url")
            if subitem_url == url.path or subitem_url == url.resolver_match.url_name:
                return True

            # Recursively check for submenus
            elif subitem.get("submenu"):
                if filter_by_url(subitem["submenu"], url):
                    return True

    return False


# Check if the user has the group
@register.filter
def has_group(user, group):
    if user.groups.filter(name=group).exists():
        return True

# Check if the user has the permission
@register.filter
def has_permission(user, permission):
    if user.has_perm(permission):
        return True


# For checking if the user group is admin
@register.filter(name="is_admin")
def is_admin(user):
    return user.groups.filter(name="admin").exists()

@register.filter(name="admin_required")
def admin_required(view_func):
    return user_passes_test(is_admin, login_url='login')(view_func)


# For checking if the user group is client
@register.filter(name="is_client")
def is_client(user):
    return user.groups.filter(name="client").exists()

@register.filter(name="client_required")
def client_required(view_func):
    return user_passes_test(is_client, login_url='login')(view_func)


# For checking if is_superuser
@register.filter(name="is_superuser")
def is_superuser(user):
    return user.is_superuser

@register.filter(name="superuser_required")
def superuser_required(view_func):
    return user_passes_test(is_superuser, login_url='login')(view_func)


# For checking if is_staff
@register.filter(name="is_staff")
def is_staff(user):
    return user.is_staff

@register.filter(name="staff_required")
def staff_required(view_func):
    return user_passes_test(is_staff, login_url='login')(view_func)

@register.simple_tag
def current_url(request):
    return request.build_absolute_uri()
