# from django.http import HttpResponse
from django.shortcuts import render
# from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
# from fl_server.forms import LoginForm, UserRegistrationForm, UserEditForm, ProfileEditForm
#from .models import Model
# from django.contrib import messages


@login_required
def dashboard(request):
    return render(request,
                  'account/dashboard.html',
                  {'section': 'Dashboard'})
