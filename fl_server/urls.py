from django.urls import path, include
from fl_server import views

app_name = "fl_server"

urlpatterns = [
    # path('', views.model_list, name='model_list'),
    path("", views.index, name="index"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("server_projects/", views.server_projects, name="projects"),
    path("server_stakeholders/", views.server_stakeholders, name="stakeholders"),
    path("server_monitoring/", views.server_monitoring, name="monitoring"),
    path("server_management/", views.server_management, name="management"),
]
