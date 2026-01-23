from django.conf import settings


def my_setting(request):
    """Add Django settings to the template context.

    Warning:
        Exposing the full ``settings`` object to templates can leak sensitive
        configuration (e.g., secrets, keys, internal endpoints). Prefer exposing
        only the specific values you actually need.

    Args:
        request: The incoming Django request object.

    Returns:
        A context dictionary containing the ``MY_SETTING`` key mapped to the
        Django settings object.

    """
    return {"MY_SETTING": settings}


def language_code(request):
    """Add the request language code to the template context.

    Args:
        request: The incoming Django request object.

    Returns:
        A context dictionary containing the current request's ``LANGUAGE_CODE``.

    """
    return {"LANGUAGE_CODE": request.LANGUAGE_CODE}


def get_cookie(request):
    """Add request cookies to the template context.

    Note:
        Cookies may contain user/session identifiers. Only expose this in
        templates if you have a clear, safe use case (typically debugging).

    Args:
        request: The incoming Django request object.

    Returns:
        A context dictionary containing the request ``COOKIES`` mapping.

    """
    return {"COOKIES": request.COOKIES}


# Add the 'ENVIRONMENT' setting to the template context
def environment(request):
    """Add the application environment name to the template context.

    This makes ``settings.ENVIRONMENT`` available in templates (e.g., to display
    a badge like "dev", "staging", or "prod").

    Args:
        request: The incoming Django request object.

    Returns:
        A context dictionary containing the ``ENVIRONMENT`` string.

    """
    return {"ENVIRONMENT": settings.ENVIRONMENT}
