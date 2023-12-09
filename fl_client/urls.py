from django.urls import path, include
from fl_client import views

app_name = 'fl_server'

urlpatterns = [
    path('', views.index, name='index')
]