from django.urls import path, include
from fl_server import views

app_name = 'fl_server'

urlpatterns = [
    # path('', views.model_list, name='model_list'),
    path('', views.dashboard, name='dashboard'),
]
