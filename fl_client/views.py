from django.shortcuts import render
from .forms import DLClassificationForm, DLSegmentation, MLClassificationForm, MLRegressionForm, MLTimeSeriesForm, \
    MLClusteringForm, MLAnomalyDetectionForm, NLPTextGenerationForm, NLPEmotionalAnalysisForm


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
