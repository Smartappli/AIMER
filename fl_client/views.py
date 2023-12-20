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
from .models import Profile, Model, Model_File, Model_Family, Model_Document, Queue
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

            # Xception #
            if cd['dpcla_Xception']:
                Queue.objects.create(queue_model_id=1, queue_state='CR')

            # VGG #
            if cd['dpcla_VGG11']:
                Queue.objects.create(queue_model_id=2, queue_state='CR')

            if cd['dpcla_VGG13']:
                Queue.objects.create(queue_model_id=3, queue_state='CR')

            if cd['dpcla_VGG16']:
                Queue.objects.create(queue_model_id=4, queue_state='CR')

            if cd['dpcla_VGG19']:
                Queue.objects.create(queue_model_id=5, queue_state='CR')

            # ResNet, ResNet V2, ResNetRS #
            if cd['dpcla_ResNet18']:
                Queue.objects.create(queue_model_id=6, queue_state='CR')

            if cd["dpcla_ResNet34"]:
                Queue.objects.create(queue_model_id=7, queue_state='CR')

            if cd["dpcla_ResNet50"]:
                Queue.objects.create(queue_model_id=8, queue_state='CR')

            if cd['dpcla_ResNet50V2']:
                Queue.objects.create(queue_model_id=9, queue_state='CR')

            if cd['dpcla_ResNetRS50']:
                Queue.objects.create(queue_model_id=10, queue_state='CR')

            if cd['dpcla_ResNet101']:
                Queue.objects.create(queue_model_id=11, queue_state='CR')

            if cd['dpcla_ResNet101V2']:
                Queue.objects.create(queue_model_id=12, queue_state='CR')

            if cd['dpcla_ResNetRS101']:
                Queue.objects.create(queue_model_id=13, queue_state='CR')

            if cd['dpcla_ResNet152']:
                Queue.objects.create(queue_model_id=14, queue_state='CR')

            if cd['dpcla_ResNet152V2']:
                Queue.objects.create(queue_model_id=15, queue_state='CR')

            if cd['dpcla_ResNetRS152']:
                Queue.objects.create(queue_model_id=16, queue_state='CR')

            if cd['dpcla_ResNetRS200']:
                Queue.objects.create(queue_model_id=17, queue_state='CR')

            if cd['dpcla_ResNetRS270']:
                Queue.objects.create(queue_model_id=18, queue_state='CR')

            if cd['dpcla_ResNetRS350']:
                Queue.objects.create(queue_model_id=19, queue_state='CR')

            if cd['dpcla_ResNetRS420']:
                Queue.objects.create(queue_model_id=20, queue_state='CR')

            # Inception
            if cd['dpcla_InceptionV3']:
                Queue.objects.create(queue_model_id=21, queue_state='CR')

            if cd['dpcla_InceptionResNetV2']:
                Queue.objects.create(queue_model_id=22, queue_state='CR')

            # MobileNet
            if cd['dpcla_MobileNet']:
                Queue.objects.create(queue_model_id=23, queue_state='CR')

            if cd['dpcla_MobileNetV2']:
                Queue.objects.create(queue_model_id=24, queue_state='CR')

            if cd['dpcla_MobileNetV3Small']:
                Queue.objects.create(queue_model_id=25, queue_state='CR')

            if cd['dpcla_MobileNetV3Large']:
                Queue.objects.create(queue_model_id=26, queue_state='CR')

            # DenseNet #
            if cd['dpcla_DenseNet121']:
                Queue.objects.create(queue_model_id=27, queue_state='CR')

            if cd['dpcla_DenseNet169']:
                Queue.objects.create(queue_model_id=28, queue_state='CR')

            if cd['dpcla_DenseNet201']:
                Queue.objects.create(queue_model_id=29, queue_state='CR')

            # NASNet #
            if cd['dpcla_NASNetMobile']:
                Queue.objects.create(queue_model_id=30, queue_state='CR')

            if cd['dpcla_NASNetLarge']:
                Queue.objects.create(queue_model_id=31, queue_state='CR')

            # EfficientNet, EfficientNet V2
            if cd['dpcla_EfficientNetB0']:
                Queue.objects.create(queue_model_id=32, queue_state='CR')

            if cd['dpcla_EfficientNetB0V2']:
                Queue.objects.create(queue_model_id=33, queue_state='CR')

            if cd['dpcla_EfficientNetB1']:
                Queue.objects.create(queue_model_id=34, queue_state='CR')

            if cd['dpcla_EfficientNetB1V2']:
                Queue.objects.create(queue_model_id=35, queue_state='CR')

            if cd['dpcla_EfficientNetB2']:
                Queue.objects.create(queue_model_id=36, queue_state='CR')

            if cd['dpcla_EfficientNetB2V2']:
                Queue.objects.create(queue_model_id=37, queue_state='CR')

            if cd['dpcla_EfficientNetB3']:
                Queue.objects.create(queue_model_id=38, queue_state='CR')

            if cd['dpcla_EfficientNetB3V2']:
                Queue.objects.create(queue_model_id=39, queue_state='CR')

            if cd['dpcla_EfficientNetB4']:
                Queue.objects.create(queue_model_id=40, queue_state='CR')

            if cd['dpcla_EfficientNetB5']:
                Queue.objects.create(queue_model_id=41, queue_state='CR')

            if cd['dpcla_EfficientNetB6']:
                Queue.objects.create(queue_model_id=42, queue_state='CR')

            if cd['dpcla_EfficientNetB7']:
                Queue.objects.create(queue_model_id=43, queue_state='CR')

            if cd['dpcla_EfficientNetV2Small']:
                Queue.objects.create(queue_model_id=44, queue_state='CR')

            if cd['dpcla_EfficientNetV2Medium']:
                Queue.objects.create(queue_model_id=45, queue_state='CR')

            if cd['dpcla_EfficientNetV2Large']:
                Queue.objects.create(queue_model_id=46, queue_state='CR')

            # ConvNeXt #
            if cd['dpcla_ConvNeXtTiny']:
                Queue.objects.create(queue_model_id=47, queue_state='CR')

            if cd['dpcla_ConvNeXtSmall']:
                Queue.objects.create(queue_model_id=48, queue_state='CR')

            if cd['dpcla_ConvNeXtBase']:
                Queue.objects.create(queue_model_id=49, queue_state='CR')

            if cd['dpcla_ConvNeXtLarge']:
                Queue.objects.create(queue_model_id=50, queue_state='CR')

            if cd['dpcla_ConvNeXtXLarge']:
                Queue.objects.create(queue_model_id=51, queue_state='CR')

            # RegNetX, RegNetY#
            if cd['dpcla_RegNetX002']:
                Queue.objects.create(queue_model_id=52, queue_state='CR')

            if cd['dpcla_RegNetY002']:
                Queue.objects.create(queue_model_id=53, queue_state='CR')

            if cd['dpcla_RegNetX004']:
                Queue.objects.create(queue_model_id=54, queue_state='CR')

            if cd['dpcla_RegNetY004']:
                Queue.objects.create(queue_model_id=55, queue_state='CR')

            if cd['dpcla_RegNetX006']:
                Queue.objects.create(queue_model_id=56, queue_state='CR')

            if cd['dpcla_RegNetY006']:
                Queue.objects.create(queue_model_id=57, queue_state='CR')

            if cd['dpcla_RegNetX008']:
                Queue.objects.create(queue_model_id=58, queue_state='CR')

            if cd['dpcla_RegNetY008']:
                Queue.objects.create(Queue_model_id=59, queue_state='CR')

            if cd['dpcla_RegNetX016']:
                Queue.objects.create(queue_model_id=60, queue_state='CR')

            if cd['dpcla_RegNetY016']:
                Queue.objects.create(Queue_model_id=61, queue_state='CR')

            if cd['dpcla_RegNetX032']:
                Queue.objects.create(queue_model_id=62, queue_state='CR')

            if cd['dpcla_RegNetY032']:
                Queue.objects.create(Queue_model_id=63, queue_state='CR')

            if cd['dpcla_RegNetX040']:
                Queue.objects.create(queue_model_id=64, queue_state='CR')

            if cd['dpcla_RegNetY40']:
                Queue.objects.create(queue_model_id=65, queue_state='CR')

            if cd['dpcla_RegNetX064']:
                Queue.objects.create(queue_model_id=66, queue_state='CR')

            if cd['dpcla_RegNetY064']:
                Queue.objects.create(queue_model_id=67, queue_state='CR')

            if cd['dpcla_RegNetX080']:
                Queue.objects.create(queue_model_id=68, queue_state='CR')

            if cd['dpcla_RegNetY80']:
                Queue.objects.create(queue_model_id=69, queue_state='CR')

            if cd['dpcla_RegNetX120']:
                Queue.objects.create(queue_model_id=70, queue_state='CR')

            if cd['dpcla_RegNetY120']:
                Queue.objects.create(queue_model_id=71, queue_state='CR')

            if cd['dpcla_RegNetX160']:
                Queue.objects.create(queue_model_id=72, queue_state='CR')

            if cd['dpcla_RegNetY160']:
                Queue.objects.create(Queue_model_id=73, queue_state='CR')

            if cd['dpcla_RegNetX320']:
                Queue.objects.create(queue_model_id=74, queue_state='CR')

            if cd['dpcla_RegNetY320']:
                Queue.objects.create(queue_model_id=75, queue_state='CR')


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
