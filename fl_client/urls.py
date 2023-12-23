"""
Copyright (C) 2024  Olivier DEBAUCHE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from django.urls import path, include
from fl_client import views

app_name = 'fl_client'

urlpatterns = [
    path('', views.index, name='index'),
    path('', include('django.contrib.auth.urls')),
    path('data_processing/', views.data_processing, name='data_processing'),
    path('data_processing_faqs/', views.data_processing_faqs, name='data_processing_faqs'),
    path('data_processing_models/', views.data_processing_models, name='data_processing_models'),
    path('data_processing_tutorials/', views.data_processing_tutorials, name='data_processing_tutorials'),
    path('deep_learning/', views.deep_learning, name='deep_learning'),
    path('deep_learning_faqs/', views.deep_learning_faqs, name='deepl_learning_faqs'),
    path('deep_learning_models/', views.deep_learning_models, name='deep_learning_models'),
    path('deep_learning_tutorials/', views.deep_learning_tutorials, name='deep_learning_tutorials'),
    path('machine_learning/', views.machine_learning, name='machine_learning'),
    path('machine_learning_faqs/', views.machine_learning_faqs, name='machine_learning_faqs'),
    path('machine_learning_models/', views.machine_learning_models, name='machine_learning_models'),
    path('machine_learning_tutorials/', views.machine_learning_tutorials, name='machine_learning_tutorials'),
    path('natural_language_processing/', views.natural_language_processing, name='natural_language_processing'),
    path('natural_language_processing_faqs/', views.natural_language_processing_faqs, name='natural_language_processing_faqs'),
    path('natural_language_processing_models/', views.natural_language_processing_models, name='natural_language_processing_models'),
    path('natural_language_processing_tutorials/', views.natural_language_processing_tutorials, name='natural_language_processing_tutorials'),
    path('register/', views.register, name='register'),
    path('edit/', views.edit, name='edit'),
    path('import/', views.import_data, name='import')
]