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
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.shortcuts import render

from .forms import DLClassificationForm, DLSegmentation
from .forms import MLClassificationForm, MLRegressionForm
from .forms import MLTimeSeriesForm
from .forms import MLClusteringForm, MLAnomalyDetectionForm
from .forms import NLPTextGenerationForm, NLPEmotionalAnalysisForm
from .forms import UserRegistrationForm, UserEditForm, ProfileEditForm
from .models import Profile, Model, Model_File, Queue  # Model_Family, Model_Document,


def index(request):
    """Method to render the index page."""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "core/index.html",
                  {"logo": logo})


def import_data(request):
    """Moethod for importing data"""
    from huggingface_hub import list_repo_tree

    my_model = Model.objects.filter(model_category='NL',
                                    model_type='TG',
                                    model_provider='HF')
    grandtotal = 0
    for p in my_model:
        the_model = Model.objects.get(pk=p.model_id)
        repo_tree = list(list_repo_tree(p.model_repo, expand=True))
        total = 0

        print(list(repo_tree))
        for fichier in repo_tree:
            q = ''
            path = ''
            file_size = 0
            sha256 = ''
            model_type = ''
            insertion = 0

            fichier_split = fichier.path.split('.')

            if (fichier_split[-1] == "gguf") or (fichier_split[-1].split('-')[0] == 'gguf'):
                q = fichier_split[-2]
                path = fichier.path
                file_size = int(fichier.lfs["size"])
                sha256 = fichier.lfs['sha256']
                insertion = 1

            match q:
                case 'Q2_K':
                    model_type = 'Q2K'

                case 'Q3_K_L':
                    model_type = 'Q3KL'

                case 'Q3_K_L':
                    model_type = 'Q3KL'

                case 'Q3_K_M':
                    model_type = 'Q3KM'

                case 'Q3_K_S':
                    model_type = 'Q3KS'

                case 'Q4_0':
                    model_type = 'Q40'

                case 'Q4_1':
                    model_type = 'Q41'

                case 'Q4_K_M':
                    model_type = 'Q4KM'

                case 'Q4_K_S':
                    model_type = 'Q4KS'

                case 'Q5_0':
                    model_type = 'Q50'

                case 'Q5_1':
                    model_type = 'Q51'

                case 'Q5_K_M':
                    model_type = 'Q5KM'

                case 'Q5_K_S':
                    model_type = 'Q5KS'

                case 'Q6_K':
                    model_type = 'Q6K'

                case 'Q8_0':
                    model_type = 'Q80'

            if insertion == 1:
                Model_File.objects.get_or_create(model_file_model_id=the_model,
                                                 model_file_type=model_type,
                                                 model_file_filename=path,
                                                 model_file_extension='GGUF',
                                                 model_file_size=file_size,
                                                 model_file_sha256=sha256
                                                 )
                total += file_size
                if model_type == 'Q4KM':
                    grandtotal += file_size

        print(p.model_repo + ": " + str(total))

    print("TOTAL: " + str(format(grandtotal / 1024 / 1024 / 1024, '.2f')) + " GB")

    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "core/index.html",
                  {"logo": logo})


def download_data(request):
    """Method to download the data from Hugging Face"""
    import os
    import shutil
    from huggingface_hub import hf_hub_download, try_to_load_from_cache, _CACHED_NO_EXIST

    my_models = Model.objects.filter(model_category='NL',
                                     model_type='TG',
                                     model_provider='HF')

    for p in my_models:
        print(p.model_repo)

        my_files = Model_File.objects.filter(model_file_model_id=p.model_id,
                                             model_file_type='Q4KM').order_by('model_file_filename')

        model_listing = []
        for q in my_files:
            filepath = try_to_load_from_cache(repo_id=p.model_repo,
                                              filename=q.model_file_filename,
                                              repo_type="model")
            if isinstance(filepath, str):
                # file exists and is cached
                print("File in cache")
                print(filepath)

            elif filepath is _CACHED_NO_EXIST:
                # non-existence of file is cached
                print("File in download")
                hf_hub_download(repo_id=p.model_repo,
                                filename=q.model_file_filename)
                print("File downloaded")

            else:
                print("File in download")

                hf_hub_download(repo_id=p.model_repo,
                                filename=q.model_file_filename)

                print("File downloaded")

            model_listing.append(try_to_load_from_cache(repo_id=p.model_repo,
                                                     filename=q.model_file_filename,
                                                     repo_type="model"))

            if len(model_listing) > 1:
                model_listing.sort()
                new_name = model_listing[0]
                target = new_name.replace('-split-a', '')

                for file in model_listing[1:]:
                    with open(new_name, 'ab') as out_file, open(file, 'rb') as in_file:
                        shutil.copyfileobj(in_file, out_file)
                        os.remove(file)

                    os.rename(new_name, target)

                i = 0
                for q2 in my_files:
                    if i == 0:
                        Model_File.objects.filter(pk=q2.model_file_model_id).update(
                            model_file_filename=q2.model_file_filename.replace('-split-a', ''))
                        i = 1
                    else:
                        Model_File.objects.get(pk=q2.model_file_model_id).delete()

    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "core/index.html",
                  {"logo": logo})


def data_processing(request):
    """Method to generate data processing form"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "data_processing/data_processing.html",
                  {"logo": logo, "section": 'data'})


def data_processing_faqs(request):
    """Method to display data faqs"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "data_processing/data_processing_faqs.html",
                  {"logo": logo, "section": 'data'})


def data_processing_models(request):
    """Method to display data processing models"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "data_processing/data_processing_models.html",
                  {"logo": logo, "section": 'data'})


def data_processing_tutorials(request):
    """Method to display data processing tutorials"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "data_processing/data_processing_tutorials.html",
                  {"logo": logo, "section": 'data'})


def deep_learning(request):
    """Method to render deep learning form"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    form1 = DLClassificationForm()
    form2 = DLSegmentation()
    return render(request,
                  "deep_learning/deep_learning.html",
                  {"logo": logo, "form1": form1, "form2": form2, "section": 'dl', "pdf": False})


def deep_learning_faqs(request):
    """Method to display deep learning faqs"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "deep_learning/deep_learning_faqs.html",
                  {"logo": logo, "section": 'dl'})


def deep_learning_models(request):
    """Method to display all deep learning models."""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "deep_learning/deep_learning_models.html",
                  {"logo": logo, "section": 'dl'})


def deep_learning_classification_run(request):
    """Method to run the deep learning classification"""
    if request.method == "POST":
        form = DLClassificationForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            model_id = 1

            # Xception #
            if cd['dpcla_Xception']:
                model_id = Model.objects.get(pk=1)

            # VGG #
            if cd['dpcla_VGG11']:
                model_id = Model.objects.get(pk=2)

            if cd['dpcla_VGG13']:
                model_id = Model.objects.get(pk=3)

            if cd['dpcla_VGG16']:
                model_id = Model.objects.get(pk=4)

            if cd['dpcla_VGG19']:
                model_id = Model.objects.get(pk=5)

            # ResNet, ResNet V2, ResNetRS #
            if cd['dpcla_ResNet18']:
                model_id = Model.objects.get(pk=6)

            if cd["dpcla_ResNet34"]:
                model_id = Model.objects.get(pk=7)

            if cd["dpcla_ResNet50"]:
                model_id = Model.objects.get(pk=8)

            if cd['dpcla_ResNet50V2']:
                model_id = Model.objects.get(pk=9)

            if cd['dpcla_ResNetRS50']:
                model_id = Model.objects.get(pk=10)

            if cd['dpcla_ResNet101']:
                model_id = Model.objects.get(pk=11)

            if cd['dpcla_ResNet101V2']:
                model_id = Model.objects.get(pk=12)

            if cd['dpcla_ResNetRS101']:
                model_id = Model.objects.get(pk=13)

            if cd['dpcla_ResNet152']:
                model_id = Model.objects.get(pk=14)

            if cd['dpcla_ResNet152V2']:
                model_id = Model.objects.get(pk=15)

            if cd['dpcla_ResNetRS152']:
                model_id = Model.objects.get(pk=16)

            if cd['dpcla_ResNetRS200']:
                model_id = Model.objects.get(pk=17)

            if cd['dpcla_ResNetRS270']:
                model_id = Model.objects.get(pk=18)

            if cd['dpcla_ResNetRS350']:
                model_id = Model.objects.get(pk=19)

            if cd['dpcla_ResNetRS420']:
                model_id = Model.objects.get(pk=20)

            # Inception
            if cd['dpcla_InceptionV3']:
                model_id = Model.objects.get(pk=21)

            if cd['dpcla_InceptionResNetV2']:
                model_id = Model.objects.get(pk=22)

            # MobileNet
            if cd['dpcla_MobileNet']:
                model_id = Model.objects.get(pk=23)

            if cd['dpcla_MobileNetV2']:
                model_id = Model.objects.get(pk=24)

            if cd['dpcla_MobileNetV3Small']:
                model_id = Model.objects.get(pk=25)

            if cd['dpcla_MobileNetV3Large']:
                model_id = Model.objects.get(pk=26)

            # DenseNet #
            if cd['dpcla_DenseNet121']:
                model_id = Model.objects.get(pk=27)

            if cd['dpcla_DenseNet169']:
                model_id = Model.objects.get(pk=28)

            if cd['dpcla_DenseNet201']:
                model_id = Model.objects.get(pk=29)

            # NASNet #
            if cd['dpcla_NASNetMobile']:
                model_id = Model.objects.get(pk=30)

            if cd['dpcla_NASNetLarge']:
                model_id = Model.objects.get(pk=31)

            # EfficientNet, EfficientNet V2
            if cd['dpcla_EfficientNetB0']:
                model_id = Model.objects.get(pk=32)

            if cd['dpcla_EfficientNetB0V2']:
                model_id = Model.objects.get(pk=33)

            if cd['dpcla_EfficientNetB1']:
                model_id = Model.objects.get(pk=34)

            if cd['dpcla_EfficientNetB1V2']:
                model_id = Model.objects.get(pk=35)

            if cd['dpcla_EfficientNetB2']:
                model_id = Model.objects.get(pk=36)

            if cd['dpcla_EfficientNetB2V2']:
                model_id = Model.objects.get(pk=37)

            if cd['dpcla_EfficientNetB3']:
                model_id = Model.objects.get(pk=38)

            if cd['dpcla_EfficientNetB3V2']:
                model_id = Model.objects.get(pk=39)

            if cd['dpcla_EfficientNetB4']:
                model_id = Model.objects.get(pk=40)

            if cd['dpcla_EfficientNetB5']:
                model_id = Model.objects.get(pk=41)

            if cd['dpcla_EfficientNetB6']:
                model_id = Model.objects.get(pk=42)

            if cd['dpcla_EfficientNetB7']:
                model_id = Model.objects.get(pk=43)

            if cd['dpcla_EfficientNetV2Small']:
                model_id = Model.objects.get(pk=44)

            if cd['dpcla_EfficientNetV2Medium']:
                model_id = Model.objects.get(pk=45)

            if cd['dpcla_EfficientNetV2Large']:
                model_id = Model.objects.get(pk=46)

            # ConvNeXt #
            if cd['dpcla_ConvNeXtTiny']:
                model_id = Model.objects.get(pk=47)

            if cd['dpcla_ConvNeXtSmall']:
                model_id = Model.objects.get(pk=48)

            if cd['dpcla_ConvNeXtBase']:
                model_id = Model.objects.get(pk=49)

            if cd['dpcla_ConvNeXtLarge']:
                model_id = Model.objects.get(pk=50)

            if cd['dpcla_ConvNeXtXLarge']:
                model_id = Model.objects.get(pk=51)

            # RegNetX, RegNetY#
            if cd['dpcla_RegNetX002']:
                model_id = Model.objects.get(pk=52)

            if cd['dpcla_RegNetY002']:
                model_id = Model.objects.get(pk=53)

            if cd['dpcla_RegNetX004']:
                model_id = Model.objects.get(pk=54)

            if cd['dpcla_RegNetY004']:
                model_id = Model.objects.get(pk=55)

            if cd['dpcla_RegNetX006']:
                model_id = Model.objects.get(pk=56)

            if cd['dpcla_RegNetY006']:
                model_id = Model.objects.get(pk=57)

            if cd['dpcla_RegNetX008']:
                model_id = Model.objects.get(pk=58)

            if cd['dpcla_RegNetY008']:
                model_id = Model.objects.get(pk=59)

            if cd['dpcla_RegNetX016']:
                model_id = Model.objects.get(pk=60)

            if cd['dpcla_RegNetY016']:
                model_id = Model.objects.get(pk=61)

            if cd['dpcla_RegNetX032']:
                model_id = Model.objects.get(pk=62)

            if cd['dpcla_RegNetY032']:
                model_id = Model.objects.get(pk=63)

            if cd['dpcla_RegNetX040']:
                model_id = Model.objects.get(pk=64)

            if cd['dpcla_RegNetY40']:
                model_id = Model.objects.get(pk=65)

            if cd['dpcla_RegNetX064']:
                model_id = Model.objects.get(pk=66)

            if cd['dpcla_RegNetY064']:
                model_id = Model.objects.get(pk=67)

            if cd['dpcla_RegNetX080']:
                model_id = Model.objects.get(pk=68)

            if cd['dpcla_RegNetY80']:
                model_id = Model.objects.get(pk=69)

            if cd['dpcla_RegNetX120']:
                model_id = Model.objects.get(pk=70)

            if cd['dpcla_RegNetY120']:
                model_id = Model.objects.get(pk=71)

            if cd['dpcla_RegNetX160']:
                model_id = Model.objects.get(pk=72)

            if cd['dpcla_RegNetY160']:
                model_id = Model.objects.get(pk=73)

            if cd['dpcla_RegNetX320']:
                model_id = Model.objects.get(pk=74)

            if cd['dpcla_RegNetY320']:
                model_id = Model.objects.get(pk=75)

            params = {}

            augmentation = {'cropping': cd['dpcla_data_augmentation_cropping'],
                            'horizontal_flip': cd['dpcla_data_augmentation_horizontal_flip'],
                            'vertical_flip': cd['dpcla_data_augmentation_vertical_flip'],
                            'translation': cd['dpcla_data_augmentation_translation'],
                            'rotation': cd['dpcla_data_augmentation_rotation'],
                            'zoom': cd['dpcla_data_augmentation_zoom'],
                            'contrast': cd['dpcla_data_augmentation_contrast'],
                            'brightness': cd['dpcla_data_augmentation_brightness']}

            params['augmentation'] = augmentation

            xai = {'activation_maximization': cd['dpcla_activationmaximization'],
                   'gradcam': cd['dpcla_gradcam'],
                   'gradcamplusplus': cd['dpcla_gradcamplusplus'],
                   'scorecam': cd['dpcla_scorecam'],
                   'fasterscorecam': cd['dpcla_fasterscorecam'],
                   'layercam': cd['dpcla_layercam'],
                   'vanillasaliency': cd['dpcla_vanillasaliency'],
                   'smoothgrad': cd['dpcla_smoothgrad']}

            params['xai'] = xai

            output = {'save_model': cd['dpcla_savemodel'],
                      'train_graph': cd['dpcla_traingraph'],
                      'confmatrix': cd['dpcla_confmatrix'],
                      'classreport': cd['dpcla_classreport'],
                      'tflite': cd['dpcla_tflite']}

            params['output'] = output

            Queue.objects.create(queue_model_id=model_id, queue_model_type='DLCL', queue_state='CR',
                                 queue_params=params)


def deep_learning_segmentation_run(request):
    """Method to run a deep learning segmentation"""
    if request.method == "POST":
        form = DLSegmentation(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            print(cd)

def deep_learning_tutorials(request):
    """Method to generate deep learning tutorials"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "deep_learning/deep_learning_tutorials.html",
                  {"logo": logo, "section": 'dl'})


def machine_learning(request):
    """Method to create a machine learning form"""
    form1 = MLClassificationForm
    form2 = MLRegressionForm
    form3 = MLTimeSeriesForm
    form4 = MLClusteringForm
    form5 = MLAnomalyDetectionForm
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "machine_learning/machine_learning.html",
                  {"logo": logo,
                   "form1": form1,
                   "form2": form2,
                   "form3": form3,
                   "form4": form4,
                   "form5": form5,
                   "section": 'ml'})


def machine_learning_anomaly_detection_run(request):
    """Method to run machine learning anomaly detection"""
    if request.method == "POST":
        form = MLAnomalyDetectionForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            print(cd)


def machine_learning_classification_run(request):
    """Method to run machine learning classification"""
    if request.method == "POST":
        form = MLClassificationForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            print(cd)


def machine_learning_clustering_run(request):
    """Method to run machine learning clustering"""
    if request.method == "POST":
        form = MLClusteringForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            print(cd)


def machine_learning_faqs(request):
    """Method to generate machine learning faqs"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "machine_learning/machine_learning_faqs.html",
                  {"logo": logo, "section": 'ml'})


def machine_learning_models(request):
    """Method to generate machine learning models list"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "machine_learning/machine_learning_models.html",
                  {"logo": logo, "section": 'ml'})


def machine_learning_regression_run(request):
    """Method for executing machine learning"""
    if request.method == "POST":
        form = MLRegressionForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            print(cd)


def machine_learning_timeseries_run(request):
    """Method for executing time series analysis"""
    if request.method == "POST":
        form = MLTimeSeriesForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            print(cd)


def machine_learning_tutorials(request):
    """Method for create machine learning tutorials page"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "machine_learning/machine_learning_tutorials.html",
                  {"logo": logo, "section": 'ml'})


def natural_language_processing(request):
    """Method for creating natural language"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    form1 = NLPTextGenerationForm()
    form2 = NLPEmotionalAnalysisForm()
    return render(request,
                  "natural_language_processing/natural_language_processing.html",
                  {"logo": logo, "form1": form1, "form2": form2, "section": 'nlp', "pdf": True})


def natural_language_processing_emotional_analysis_run(request):
    """Method for execute natural language processing emotional analysis"""
    if request.method == "POST":
        form = NLPEmotionalAnalysisForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            print(cd)


def natural_language_processing_faqs(request):
    """Method to display natural language processing faqs"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "natural_language_processing/natural_language_processing_faqs.html",
                  {"logo": logo, "section": 'nlp'})


def natural_language_processing_models(request):
    """Method to display all models available"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "natural_language_processing/natural_language_processing_models.html",
                  {"logo": logo, "section": 'nlp'})


def natural_language_processing_text_generation_run(request):
    """Method to use text generation natural language processing models"""
    if request.method == "POST":
        form = NLPTextGenerationForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            print(cd)


def natural_language_processing_tutorials(request):
    """Method to generate natural language processing tutorials"""
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request,
                  "natural_language_processing/natural_language_processing_tutorials.html",
                  {"logo": logo, "section": 'nlp'})


def model_list(request):
    """Method to list all models"""
    models = Model.objects.all()
    return render(request,
                  'base.html',
                  {'models': models})


def register(request):
    """Method to register a new model"""
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
    """Method to edit an existing user profile"""
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
