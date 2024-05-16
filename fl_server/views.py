# from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import render

# from django.contrib.auth import authenticate, login
# from django.contrib.auth.decorators import login_required
# from fl_server.forms import LoginForm, UserRegistrationForm, UserEditForm, ProfileEditForm
# from .models import Model
# from django.contrib import messages


def index(request):
    """Class method that create main page"""
    return HttpResponseRedirect("/server/dashboard/")


def dashboard(request):
    """Class method that create the main page"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(request, "server/server_dashboard.html",
                  {"section": "Dashboard", "logo": logo})


def server_projects(request):
    """Class method that create projects page"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(request, "server/server_projects.html",
                  {"section": "projects", "logo": logo})


def server_stakeholders(request):
    """Class method that create stakejolders page"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "server/server_stakeholders.html",
        {"section": "stakeholders", "logo": logo},
    )


def server_monitoring(request):
    """Class method that create monitoring page"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "server/server_monitoring.html",
        {"section": "monitoring", "logo": logo},
    )


def server_management(request):
    """Class method that create the server management page"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "server/server_management.html",
        {"section": "management", "logo": logo},
    )
