from django.conf import settings

def my_setting(request):
    return {'MY_SETTING': settings}

def language_code(request):
    return {"LANGUAGE_CODE": request.LANGUAGE_CODE}

def get_cookie(request):
    return {"COOKIES": request.COOKIES}

# Add the 'ENVIRONMENT' setting to the template context
def environment(request):
    return {'ENVIRONMENT': settings.ENVIRONMENT}
