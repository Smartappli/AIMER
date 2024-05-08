import torch.nn as nn
from torchvision import models
from timm import create_model


def get_densenet_model(densenet_type, num_classes):
    """
    Retrieve a DenseNet model with the specified architecture and number of classes.

    Args:
        densenet_type (str): Type of DenseNet architecture to retrieve.
            Options include: 'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201',
            'densenet121', 'densenetblur121d', 'densenet169', 'densenet201', 'densenet161', 'densenet264d'.
        num_classes (int): Number of output classes for the classification layer.

    Returns:
        torch.nn.Module: DenseNet model with the specified architecture and modified classifier layer
        to match the number of classes.

    Raises:
        ValueError: If an unknown DenseNet architecture is provided.
    """
    torch_vision = False
    # Load the pre-trained version of DenseNet
    if densenet_type == 'DenseNet121':
        torch_vision = True
        try:
            weights = models.DenseNet121_Weights.DEFAULT
            densenet_model = models.densenet121(weights=weights)
        except RuntimeError:
            densenet_model = models.densenet121(weights=None)
    elif densenet_type == 'DenseNet161':
        torch_vision = True
        try:
            weights = models.DenseNet161_Weights.DEFAULT
            densenet_model = models.densenet161(weights=weights)
        except RuntimeError:
            densenet_model = models.densenet161(weights=None)
    elif densenet_type == 'DenseNet169':
        torch_vision = True
        try:
            weights = models.DenseNet169_Weights.DEFAULT
            densenet_model = models.densenet169(weights=weights)
        except RuntimeError:
            densenet_model = models.densenet169(weights=None)
    elif densenet_type == 'DenseNet201':
        torch_vision = True
        try:
            weights = models.DenseNet201_Weights.DEFAULT
            densenet_model = models.densenet201(weights=weights)
        except RuntimeError:
            densenet_model = models.densenet201(weights=None)
    elif densenet_type == "densenet121":
        try:
            densenet_model = create_model('densenet121',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            densenet_model = create_model('densenet121',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif densenet_type == "densenetblur121d":
        try:
            densenet_model = create_model('densenetblur121d',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            densenet_model = create_model('densenetblur121d',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif densenet_type == "densenet169":
        try:
            densenet_model = create_model('densenet169',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            densenet_model = create_model('densenet169',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif densenet_type == "densenet201":
        try:
            densenet_model = create_model('densenet201',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            densenet_model = create_model('densenet201',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif densenet_type == "densenet161":
        try:
            densenet_model = create_model('densenet161',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            densenet_model = create_model('densenet161',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif densenet_type == "densenet264d":
        try:
            densenet_model = create_model('densenet264d',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            densenet_model = create_model('densenet264d',
                                          pretrained=False,
                                          num_classes=num_classes)
    else:
        raise ValueError(f'Unknown DenseNet Architecture : {densenet_type}')

    if torch_vision:
        # Modify last layer to suit number of classes
        num_features = densenet_model.classifier.in_features
        densenet_model.classifier = nn.Linear(num_features, num_classes)

    return densenet_model
