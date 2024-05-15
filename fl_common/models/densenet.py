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
    # Mapping of vision types to their corresponding torchvision models and
    # weights
    torchvision_models = {
        'DenseNet121': (models.densenet121, models.DenseNet121_Weights),
        'DenseNet161': (models.densenet161, models.DenseNet161_Weights),
        'DenseNet169': (models.densenet169, models.DenseNet169_Weights),
        'DenseNet201': (models.densenet201, models.DenseNet201_Weights)
    }

    timm_models = [
        "densenet121",
        "densenetblur121d",
        "densenet169",
        "densenet201",
        "densenet161",
        "densenet264d"]

    # Check if the vision type is from torchvision
    if densenet_type in torchvision_models:
        model_func, weights_class = torchvision_models[densenet_type]
        try:
            weights = weights_class.DEFAULT
            densenet_model = model_func(weights=weights)
        except RuntimeError as e:
            print(f"{densenet_type} - Error loading pretrained model: {e}")
            densenet_model = model_func(weights=None)

        # Modify last layer to suit number of classes
        num_features = densenet_model.classifier.in_features
        densenet_model.classifier = nn.Linear(num_features, num_classes)

    # Check if the vision type is from the 'timm' library
    elif densenet_type in timm_models:
        try:
            densenet_model = create_model(
                densenet_type, pretrained=True, num_classes=num_classes)
        except RuntimeError as e:
            print(f"{densenet_type} - Error loading pretrained model: {e}")
            densenet_model = create_model(
                densenet_type, pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown DenseNet Architecture : {densenet_type}')

    return densenet_model
