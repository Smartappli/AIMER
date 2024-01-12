import torch.nn as nn
from torchvision import models


def get_resnet_model(resnet_type, num_classes):
    """
    Returns a modified ResNet model based on the specified type.

    Parameters:
        - resnet_type (str): Type of ResNet architecture.
                           Currently supports 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', and 'ResNet152'.
        - num_classes (int): Number of classes for the modified last layer.

    Returns:
        - torch.nn.Module: Modified ResNet model with the specified number of classes.

    Raises:
        - ValueError: If an unknown ResNet architecture is provided.
    """
    # Load the pre-trained version of ResNet based on the specified type
    if resnet_type == 'ResNet18':
        weights = models.ResNet18_Weights.DEFAULT
        resnet_model = models.resnet18(weights=weights)
    elif resnet_type == 'ResNet34':
        weights = models.ResNet34_Weights.DEFAULT
        resnet_model = models.resnet34(weights=weights)
    elif resnet_type == 'ResNet50':
        weights = models.ResNet50_Weights.DEFAULT
        resnet_model = models.resnet50(weights=weights)
    elif resnet_type == 'ResNet101':
        weights = models.ResNet101_Weights.DEFAULT
        resnet_model = models.resnet101(weights=weights)
    elif resnet_type == 'ResNet152':
        weights = models.ResNet152_Weights.DEFAULT
        resnet_model = models.resnet152(weights=weights)
    else:
        raise ValueError(f'Unknown ResNet Architecture: {resnet_type}')

    # Modify the last layer to suit the given number of classes
    num_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_features, num_classes)

    return resnet_model
