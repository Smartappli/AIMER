from django.conf import settings
from django.utils.translation import activate


class DefaultLanguageMiddleware:
    """Django middleware that ensures a default language is activated and persisted.

    If the ``django_language`` cookie is missing, this middleware:
    - activates ``settings.LANGUAGE_CODE`` for the current request
    - sets the ``django_language`` cookie on the response

    If the cookie is already present, the request/response cycle is passed through
    unchanged.

    Notes:
        - This middleware relies on Django's i18n machinery (``activate``) and the
          ``settings.LANGUAGE_CODE`` value.
        - Cookie name follows Django's default language cookie convention.

    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Check if the django_language cookie is not set
        if "django_language" not in request.COOKIES:
            # Get the default language from settings.LANGUAGE_CODE
            default_language = settings.LANGUAGE_CODE
            activate(default_language)
            response = self.get_response(request)
            response.set_cookie("django_language", default_language)

        else:
            response = self.get_response(request)

        return response
