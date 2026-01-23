from django.urls import path

from .views import FrontPagesView

urlpatterns = [
    path("", FrontPagesView.as_view(template_name="landing_page.html"), name="index"),
]
