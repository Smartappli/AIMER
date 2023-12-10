from django.shortcuts import render


def index(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "core/index.html", {"logo": logo})


def data_processing(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "deep_learning/deep_learning.html", {"logo": logo, "section": 'data'})


def deep_learning(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "deep_learning/deep_learning.html", {"logo": logo, "section": 'dl'})


def machine_learning(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "machine_learning/machine_learning.html", {"logo": logo, "section": 'ml'})


def natural_language_processing(request):
    logo = ['share', 'hospital', 'data', 'cpu', 'gpu']
    return render(request, "machine_learning/machine_learning.html", {"logo": logo, "section": 'nlp'})
