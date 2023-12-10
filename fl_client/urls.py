from django.urls import path, include
from fl_client import views

app_name = 'fl_client'

urlpatterns = [
    path('', views.index, name='index'),
    path('', include('django.contrib.auth.urls')),
    path('data_processing/', views.data_processing, name='data_processing'),
    path('machine_learning/', views.machine_learning, name='machine_learning'),
    path('deep_learning/', views.deep_learning, name='deepl_learning'),
    path('natural_language_processing/', views.natural_language_processing, name='natural_language_processing')
]