# from django.http import HttpResponse
from django.shortcuts import render
# from django.contrib.auth import authenticate, login
# from django.contrib.auth.decorators import login_required
# from fl_server.forms import LoginForm, UserRegistrationForm, UserEditForm, ProfileEditForm
# from .models import Model
# from django.contrib import messages


def dashboard(request):
    return render(request,
                  'server/server_dashboard.html',
                  {'section': 'Dashboard'})


def server_projects(request):
    return render(request,
                  'server/server_projects.html',
                  {'section': 'projects'})


def server_stakeholders(request):
    return render(request,
                  'server/server_stakeholders.html',
                  {'section': 'stakeholders'})



def server_monitoring(request):
    return render(request,
                  'server/server_monitoring.html',
                  {'section': 'monitoring'})


def server_management(request):
    return render(request,
                  'server/server_management.html',
                  {'section': 'management'})