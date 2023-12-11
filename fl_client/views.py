from django.shortcuts import render


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


def deep_learning_tutorials(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "deep_learning/deep_learning_tutorials.html", {"logo": logo, "section": 'dl'})


def machine_learning(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "machine_learning/machine_learning.html", {"logo": logo, "section": 'ml'})


def machine_learning_faqs(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "machine_learning/machine_learning_faqs.html", {"logo": logo, "section": 'ml'})


def machine_learning_models(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "machine_learning/machine_learning_models.html", {"logo": logo, "section": 'ml'})


def machine_learning_tutorials(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "machine_learning/machine_learning_tutorials.html", {"logo": logo, "section": 'ml'})


def natural_language_processing(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "natural_language_processing/natural_language_processing.html",
                  {"logo": logo, "section": 'nlp'})


def natural_language_processing_faqs(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "natural_language_processing/natural_language_processing_faqs.html",
                  {"logo": logo, "section": 'nlp'})


def natural_language_processing_models(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "natural_language_processing/natural_language_processing_models.html",
                  {"logo": logo, "section": 'nlp'})


def natural_language_processing_tutorials(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "natural_language_processing/natural_language_processing_tutorials.html",
                  {"logo": logo, "section": 'nlp'})