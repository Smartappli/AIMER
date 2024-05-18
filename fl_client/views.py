from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.shortcuts import render

from .forms import DLClassificationForm, DLSegmentation
from .forms import MLClassificationForm, MLRegressionForm
from .forms import MLTimeSeriesForm
from .forms import MLClusteringForm, MLAnomalyDetectionForm
from .forms import NLPTextGenerationForm, NLPEmotionalAnalysisForm
from .forms import UserRegistrationForm, UserEditForm, ProfileEditForm
from .models import Profile, Model, ModelFile, Queue  # ModelFamily, ModelDocument,

# from fl_common.models.xception import xception
# from fl_common.models.alexnet import alexnet


def index(request):
    """Method to render the index page."""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(request, "core/index.html", {"logo": logo})


def import_data(request):
    """Method for importing data"""
    from huggingface_hub import list_repo_tree

    my_model = Model.objects.filter(
        model_category="NL", model_type="TG", model_provider="HF"
    )
    grandtotal = 0
    for p in my_model:
        the_model = Model.objects.get(pk=p.model_id)
        repo_tree = list(list_repo_tree(p.model_repo, expand=True))
        total = 0

        print(list(repo_tree))
        for fichier in repo_tree:
            q = ""
            path = ""
            file_size = 0
            sha256 = ""
            model_type = ""
            insertion = 0

            fichier_split = fichier.path.split(".")

            if (fichier_split[-1] == "gguf") or (
                fichier_split[-1].split("-")[0] == "gguf"
            ):
                q = fichier_split[-2]
                path = fichier.path
                file_size = int(fichier.lfs["size"])
                sha256 = fichier.lfs["sha256"]
                insertion = 1

            match q:
                case "Q2_K":
                    model_type = "Q2K"

                case "Q3_K_L":
                    model_type = "Q3KL"

                case "Q3_K_L":
                    model_type = "Q3KL"

                case "Q3_K_M":
                    model_type = "Q3KM"

                case "Q3_K_S":
                    model_type = "Q3KS"

                case "Q4_0":
                    model_type = "Q40"

                case "Q4_1":
                    model_type = "Q41"

                case "Q4_K_M":
                    model_type = "Q4KM"

                case "Q4_K_S":
                    model_type = "Q4KS"

                case "Q5_0":
                    model_type = "Q50"

                case "Q5_1":
                    model_type = "Q51"

                case "Q5_K_M":
                    model_type = "Q5KM"

                case "Q5_K_S":
                    model_type = "Q5KS"

                case "Q6_K":
                    model_type = "Q6K"

                case "Q8_0":
                    model_type = "Q80"

            if insertion == 1:
                ModelFile.objects.get_or_create(
                    model_file_model_id=the_model,
                    model_file_type=model_type,
                    model_file_filename=path,
                    model_file_extension="GGUF",
                    model_file_size=file_size,
                    model_file_sha256=sha256,
                )
                total += file_size
                if model_type == "Q4KM":
                    grandtotal += file_size

        print(p.model_repo + ": " + str(total))

    print("TOTAL: " + str(format(grandtotal / 1024 / 1024 / 1024, ".2f")) + " GB")

    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(request, "core/index.html", {"logo": logo})


def download_data(request):
    """Method to download the data from Hugging Face"""
    import os
    import shutil
    from huggingface_hub import (
        hf_hub_download,
        try_to_load_from_cache,
        _CACHED_NO_EXIST,
    )

    my_models = Model.objects.filter(
        model_category="NL", model_type="TG", model_provider="HF"
    )

    for p in my_models:
        print(p.model_repo)

        my_files = ModelFile.objects.filter(
            model_file_model_id=p.model_id, model_file_type="Q4KM"
        ).order_by("model_file_filename")

        model_listing = []
        for q in my_files:
            filepath = try_to_load_from_cache(
                repo_id=p.model_repo, filename=q.model_file_filename, repo_type="model"
            )
            if isinstance(filepath, str):
                # file exists and is cached
                print("File in cache")
                print(filepath)

            elif filepath is _CACHED_NO_EXIST:
                # non-existence of file is cached
                print("File in download")
                hf_hub_download(repo_id=p.model_repo, filename=q.model_file_filename)
                print("File downloaded")

            else:
                print("File in download")

                hf_hub_download(repo_id=p.model_repo, filename=q.model_file_filename)

                print("File downloaded")

            model_listing.append(
                try_to_load_from_cache(
                    repo_id=p.model_repo,
                    filename=q.model_file_filename,
                    repo_type="model",
                )
            )

            if len(model_listing) > 1:
                model_listing.sort()
                new_name = model_listing[0]
                target = new_name.replace("-split-a", "")

                for file in model_listing[1:]:
                    with open(new_name, "ab") as out_file, open(file, "rb") as in_file:
                        shutil.copyfileobj(in_file, out_file)
                        os.remove(file)

                    os.rename(new_name, target)

                i = 0
                for q2 in my_files:
                    if i == 0:
                        ModelFile.objects.filter(pk=q2.model_file_model_id).update(
                            model_file_filename=q2.model_file_filename.replace(
                                "-split-a", ""
                            )
                        )
                        i = 1
                    else:
                        ModelFile.objects.get(pk=q2.model_file_model_id).delete()

    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(request, "core/index.html", {"logo": logo})


def data_processing(request):
    """Method to generate data processing form"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "data_processing/data_processing.html",
        {"logo": logo, "section": "data"},
    )


def data_processing_faqs(request):
    """Method to display data faqs"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "data_processing/data_processing_faqs.html",
        {"logo": logo, "section": "data"},
    )


def data_processing_models(request):
    """Method to display data processing models"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "data_processing/data_processing_models.html",
        {"logo": logo, "section": "data"},
    )


def data_processing_tutorials(request):
    """Method to display data processing tutorials"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "data_processing/data_processing_tutorials.html",
        {"logo": logo, "section": "data"},
    )


def deep_learning(request):
    """Method to render deep learning form"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    form1 = DLClassificationForm()
    form2 = DLSegmentation()
    optimizers = (
        "SGD",
        "RMSProp",
        "Adam",
        "AdamW",
        "Adadelta",
        "Adagrad",
        "Adamax",
        "Adafactor",
        "Nadam",
        "Ftrl",
    )
    losses = (
        "BinaryCrossentropy",
        "CategoricalCrossentropy",
        "SparseCategoricalCrossentropy",
        "Poisson",
        "KLDivergence",
        "MeanSquaredError",
        "MeanAbsoluteError",
        "MeanAbsolutePercentageError",
        "MeanSquaredLogarithmicError",
        "CosineSimilarity",
        "Huber",
        "LogCosh",
        "Hinge",
        "SquaredHinge",
        "CategoricalHinge",
    )
    lrs = (0.1, 0.01, 0.001, 0.0001, 0.00001)
    return render(
        request,
        "deep_learning/deep_learning.html",
        {
            "logo": logo,
            "form1": form1,
            "optimizers": optimizers,
            "losses": losses,
            "lrs": lrs,
            "form2": form2,
            "section": "dl",
            "pdf": False,
        },
    )


def deep_learning_faqs(request):
    """Method to display deep learning faqs"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "deep_learning/deep_learning_faqs.html",
        {"logo": logo, "section": "dl"},
    )


def deep_learning_models(request):
    """Method to display all deep learning models."""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "deep_learning/deep_learning_models.html",
        {"logo": logo, "section": "dl"},
    )


def deep_learning_classification_run(request):
    """Method to run the deep learning classification"""
    if request.method == "POST":
        form = DLClassificationForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data

            for model_key in cd.keys():
                model_name = model_key.split("_")

                try:
                    model_id = Model.objects.get(model_short_name=model_name[1])
                except Model.DoesNotExist:
                    model_id = None

                if model_id is not None:
                    params = {}

                    augmentation = {
                        "cropping": cd["dpcla_data_augmentation_cropping"],
                        "horizontal_flip": cd[
                            "dpcla_data_augmentation_horizontal_flip"
                        ],
                        "vertical_flip": cd["dpcla_data_augmentation_vertical_flip"],
                        "translation": cd["dpcla_data_augmentation_translation"],
                        "rotation": cd["dpcla_data_augmentation_rotation"],
                        "zoom": cd["dpcla_data_augmentation_zoom"],
                        "contrast": cd["dpcla_data_augmentation_contrast"],
                        "brightness": cd["dpcla_data_augmentation_brightness"],
                    }

                    params["augmentation"] = augmentation

                    xai = {
                        "activation_maximization": cd["dpcla_activationmaximization"],
                        "gradcam": cd["dpcla_gradcam"],
                        "gradcamplusplus": cd["dpcla_gradcamplusplus"],
                        "scorecam": cd["dpcla_scorecam"],
                        "fasterscorecam": cd["dpcla_fasterscorecam"],
                        "layercam": cd["dpcla_layercam"],
                        "vanillasaliency": cd["dpcla_vanillasaliency"],
                        "smoothgrad": cd["dpcla_smoothgrad"],
                    }

                    params["xai"] = xai

                    output = {
                        "save_model": cd["dpcla_savemodel"],
                        "train_graph": cd["dpcla_traingraph"],
                        "confmatrix": cd["dpcla_confmatrix"],
                        "classreport": cd["dpcla_classreport"],
                        "tflite": cd["dpcla_tflite"],
                    }

                    params["output"] = output

                    Queue.objects.create(
                        queue_model_id=model_id,
                        queue_model_type="DLCL",
                        queue_state="CR",
                        queue_params=params,
                    )


def deep_learning_segmentation_run(request):
    """Method to run a deep learning segmentation"""
    if request.method == "POST":
        form = DLSegmentation(request.POST)
        if form.is_valid():
            cd = form.cleaned_data

            model_id = 1
            if cd["dpseg_unet"]:
                model_id = Model.objects.get(pk=67)

            if cd["dpseg_unetplusplus"]:
                model_id = Model.objects.get(pk=67)

            if cd["dpseg_manet"]:
                model_id = Model.objects.get(pk=67)

            if cd["dpseg_linknet"]:
                model_id = Model.objects.get(pk=67)

            if cd["dpseg_fpn"]:
                model_id = Model.objects.get(pk=67)

            if cd["dpseg_pspnet"]:
                model_id = Model.objects.get(pk=67)

            if cd["dpseg_pan"]:
                model_id = Model.objects.get(pk=67)

            if cd["dpseg_deeplabv3"]:
                model_id = Model.objects.get(pk=67)

            if cd["dpseg_deeplabv3plus"]:
                model_id = Model.objects.get(pk=67)

            print(model_id)


def deep_learning_tutorials(request):
    """Method to generate deep learning tutorials"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "deep_learning/deep_learning_tutorials.html",
        {"logo": logo, "section": "dl"},
    )


def machine_learning(request):
    """Method to create a machine learning form"""
    form1 = MLClassificationForm
    form2 = MLRegressionForm
    form3 = MLTimeSeriesForm
    form4 = MLClusteringForm
    form5 = MLAnomalyDetectionForm
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "machine_learning/machine_learning.html",
        {
            "logo": logo,
            "form1": form1,
            "form2": form2,
            "form3": form3,
            "form4": form4,
            "form5": form5,
            "section": "ml",
        },
    )


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
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "machine_learning/machine_learning_faqs.html",
        {"logo": logo, "section": "ml"},
    )


def machine_learning_models(request):
    """Method to generate machine learning models list"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "machine_learning/machine_learning_models.html",
        {"logo": logo, "section": "ml"},
    )


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
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "machine_learning/machine_learning_tutorials.html",
        {"logo": logo, "section": "ml"},
    )


def natural_language_processing(request):
    """Method for creating natural language"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    form1 = NLPTextGenerationForm()
    form2 = NLPEmotionalAnalysisForm()
    return render(
        request,
        "natural_language_processing/natural_language_processing.html",
        {"logo": logo, "form1": form1, "form2": form2, "section": "nlp", "pdf": True},
    )


def natural_language_processing_emotional_analysis_run(request):
    """Method for execute natural language processing emotional analysis"""
    if request.method == "POST":
        form = NLPEmotionalAnalysisForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            print(cd)


def natural_language_processing_faqs(request):
    """Method to display natural language processing faqs"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "natural_language_processing/natural_language_processing_faqs.html",
        {"logo": logo, "section": "nlp"},
    )


def natural_language_processing_models(request):
    """Method to display all models available"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "natural_language_processing/natural_language_processing_models.html",
        {"logo": logo, "section": "nlp"},
    )


def natural_language_processing_text_generation_run(request):
    """Method to use text generation natural language processing models"""
    if request.method == "POST":
        form = NLPTextGenerationForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            print(cd)


def natural_language_processing_tutorials(request):
    """Method to generate natural language processing tutorials"""
    logo = ["share", "hospital", "data", "cpu", "gpu"]
    return render(
        request,
        "natural_language_processing/natural_language_processing_tutorials.html",
        {"logo": logo, "section": "nlp"},
    )


def model_list(request):
    """Method to list all models"""
    models = Model.objects.all()
    return render(request, "base.html", {"models": models})


def register(request):
    """Method to register a new model"""
    if request.method == "POST":
        user_form = UserRegistrationForm(request.POST)
        if user_form.is_valid():
            # Create a new user object but avoid saving it yet
            new_user = user_form.save(commit=False)
            # Set the chosen password
            new_user.set_password(user_form.cleaned_data["password"])
            # Save the User object
            new_user.save()
            # Create the user profile
            Profile.objects.create(user=new_user)
            return render(request, "account/register_done.html", {"new_user": new_user})

    else:
        user_form = UserRegistrationForm()
    return render(request, "account/register.html", {"user_form": user_form})


@login_required
def edit(request):
    """Method to edit an existing user profile"""
    if request.method == "POST":
        user_form = UserEditForm(instance=request.user, data=request.POST)
        profile_form = ProfileEditForm(
            instance=request.user.profile, date=request.POST, files=request.FILES
        )
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, "Profile updated successfully.")
        else:
            messages.error(request, "Error updating your profile.")
    else:
        user_form = UserEditForm(instance=request.user)
        profile_form = ProfileEditForm(instance=request.user.profile)

    return render(
        request,
        "account/edit.html",
        {"user_form": user_form, "profile_form": profile_form},
    )
