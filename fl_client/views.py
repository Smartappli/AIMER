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

from django.shortcuts import render
from .forms import (DLClassificationForm, DLSegmentation, MLClassificationForm, MLRegressionForm, MLTimeSeriesForm,
                    MLClusteringForm, MLAnomalyDetectionForm, NLPTextGenerationForm, NLPEmotionalAnalysisForm,
                    UserRegistrationForm, UserEditForm, ProfileEditForm)
from .models import Profile, Model, Model_File, Model_Family
from django.contrib.auth.decorators import login_required
from django.contrib import messages


def index(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "core/index.html", {"logo": logo})


def data_processing(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "data_processing/data_processing.html", {"logo": logo, "section": 'data'})


def data_processing_faqs(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "data_processing/data_processing_faqs.html", {"logo": logo, "section": 'data'})


def data_processing_models(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "data_processing/data_processing_models.html", {"logo": logo, "section": 'data'})


def data_processing_tutorials(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "data_processing/data_processing_tutorials.html", {"logo": logo, "section": 'data'})


def deep_learning(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "deep_learning/deep_learning.html", {"logo": logo, "section": 'dl'})


def deep_learning_faqs(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "deep_learning/deep_learning_faqs.html", {"logo": logo, "section": 'dl'})


def deep_learning_models(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "deep_learning/deep_learning_models.html", {"logo": logo, "section": 'dl'})


def deep_learning_classification_run(request):
    if request.method == "POST":
        form = DLClassificationForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data


def deep_learning_segmentation_run(request):
    if request.method == "POST":
        form = DLSegmentation(request.POST)
        if form.is_valid():
            cd = form.cleaned_data


def deep_learning_tutorials(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "deep_learning/deep_learning_tutorials.html", {"logo": logo, "section": 'dl'})


def machine_learning(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "machine_learning/machine_learning.html", {"logo": logo, "section": 'ml'})


def machine_learning_anomaly_detection_run(request):
    if request.method == "POST":
        form = MLAnomalyDetectionForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data


def machine_learning_classification_run(request):
    if request.method == "POST":
        form = MLClassificationForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data


def machine_learning_clustering_run(request):
    if request.method == "POST":
        form = MLClusteringForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data


def machine_learning_faqs(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "machine_learning/machine_learning_faqs.html", {"logo": logo, "section": 'ml'})


def machine_learning_models(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "machine_learning/machine_learning_models.html", {"logo": logo, "section": 'ml'})


def machine_learning_regression_run(request):
    if request.method == "POST":
        form = MLRegressionForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data


def machine_learning_timeseries_run(request):
    if request.method == "POST":
        form = MLTimeSeriesForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data


def machine_learning_tutorials(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "machine_learning/machine_learning_tutorials.html", {"logo": logo, "section": 'ml'})


def natural_language_processing(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "natural_language_processing/natural_language_processing.html",
                  {"logo": logo, "section": 'nlp'})


def natural_language_processing_emotional_analysis_run(request):
    if request.method == "POST":
        form = NLPEmotionalAnalysisForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data


def natural_language_processing_faqs(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "natural_language_processing/natural_language_processing_faqs.html",
                  {"logo": logo, "section": 'nlp'})


def natural_language_processing_models(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "natural_language_processing/natural_language_processing_models.html",
                  {"logo": logo, "section": 'nlp'})


def natural_language_processing_text_generation_run(request):
    if request.method == "POST":
        form = NLPTextGenerationForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data


def natural_language_processing_tutorials(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "natural_language_processing/natural_language_processing_tutorials.html",
                  {"logo": logo, "section": 'nlp'})


def model_list(request):
    models = Model.objects.all()
    return render(request,
                  'base.html',
                  {'models': models})


def register(request):
    if request.method == 'POST':
        user_form = UserRegistrationForm(request.POST)
        if user_form.is_valid():
            # Create a new user object but avoid saving it yet
            new_user = user_form.save(commit=False)
            # Set the chosen password
            new_user.set_password(
                user_form.cleaned_data['password']
            )
            # Save the User object
            new_user.save()
            # Create the user profile
            Profile.objects.create(user=new_user)
            return render(request,
                          'account/register_done.html',
                          {'new_user': new_user})

    else:
        user_form = UserRegistrationForm()
    return render(request,
                  'account/register.html',
                  {'user_form': user_form})


@login_required
def edit(request):
    if request.method == 'POST':
        user_form = UserEditForm(
            instance=request.user,
            data=request.POST)
        profile_form = ProfileEditForm(
            instance=request.user.profile,
            date=request.POST,
            files=request.FILES)
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, "Profile updated successfully.")
        else:
            messages.error(request, 'Error updating your profile.')
    else:
        user_form = UserEditForm(instance=request.user)
        profile_form = ProfileEditForm(instance=request.user.profile)

    return render(request,
                  'account/edit.html',
                  {'user_form': user_form,
                   'profile_form': profile_form})
