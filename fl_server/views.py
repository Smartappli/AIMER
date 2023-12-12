# from django.http import HttpResponse
from django.shortcuts import render
# from django.contrib.auth import authenticate, login
# from django.contrib.auth.decorators import login_required
# from fl_server.forms import LoginForm, UserRegistrationForm, UserEditForm, ProfileEditForm
# from .models import Model
# from django.contrib import messages


def dashboard(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  'server/server_dashboard.html',
                  {'section': 'Dashboard', 'logo': logo})


def server_projects(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  'server/server_projects.html',
                  {'section': 'projects', 'logo': logo})


def server_stakeholders(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  'server/server_stakeholders.html',
                  {'section': 'stakeholders', 'logo': logo})


def server_monitoring(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  'server/server_monitoring.html',
                  {'section': 'monitoring', 'logo': logo})


def server_management(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  'server/server_management.html',
                  {'section': 'management', 'logo': logo})