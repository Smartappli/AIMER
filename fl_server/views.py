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

# from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import render
# from django.contrib.auth import authenticate, login
# from django.contrib.auth.decorators import login_required
# from fl_server.forms import LoginForm, UserRegistrationForm, UserEditForm, ProfileEditForm
# from .models import Model
# from django.contrib import messages


def index(request):
    return HttpResponseRedirect("/server/dashboard/")


def dashboard(request):
    """Class method that create the main page"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  'server/server_dashboard.html',
                  {'section': 'Dashboard', 'logo': logo})


def server_projects(request):
    """Class method that create projects page"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  'server/server_projects.html',
                  {'section': 'projects', 'logo': logo})


def server_stakeholders(request):
    """Class method that create stakejolders page"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  'server/server_stakeholders.html',
                  {'section': 'stakeholders', 'logo': logo})


def server_monitoring(request):
    """Class method that create monitoring page"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  'server/server_monitoring.html',
                  {'section': 'monitoring', 'logo': logo})


def server_management(request):
    """Class method that create the server management page"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  'server/server_management.html',
                  {'section': 'management', 'logo': logo})

