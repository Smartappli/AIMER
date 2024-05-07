import torch.nn as nn
from torchvision import models


def get_googlenet_model(googlenet_type, num_classes):
    """
    Returns a modified GoogleNet model based on the specified type.

    Parameters:
        - googlenet_type (str): Type of GoogleNet architecture. Currently supports 'AlexNet'.
        - num_classes (int): Number of classes for the modified last layer.

    Returns:
        - torch.nn.Module: Modified GoogleNet model with the specified number of classes.

    Raises:
        - ValueError: If an unknown AlexNet architecture is provided.
    """
    # Load the pre-trained version of GoogleNet based on the specified type
    if googlenet_type == 'GoogLeNet':
        try:
            weights = models.GoogLeNet_Weights.DEFAULT
            googlenet_model = models.googlenet(weights=weights)
        except Exception:
            googlenet_model = models.googlenet(weights=None)
    else:
        raise ValueError(f'Unknown AlexNet Architecture: {googlenet_type}')

    # Modify the last layer to suit the given number of classes
    num_features = googlenet_model.fc.in_features
    googlenet_model.fc = nn.Linear(num_features, num_classes)

    return googlenet_model
