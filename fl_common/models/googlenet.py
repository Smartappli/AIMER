import torch.nn as nn
from torchvision import models


def get_googlenet_model(googlenet_type, num_classes):
    """
    Load a pre-trained GoogLeNet model of the specified type and modify its
    last layer to accommodate the given number of classes.

    Parameters:
    - googlenet_type (str): Type of GoogLeNet architecture, supported type:
        - 'GoogLeNet'
    - num_classes (int): Number of output classes for the modified last layer.

    Returns:
    - torch.nn.Module: Modified GoogLeNet model with the specified architecture
      and last layer adapted for the given number of classes.

    Raises:
    - ValueError: If an unknown GoogLeNet architecture type is provided.
    """
    # Validate the googlenet_type before proceeding
    if googlenet_type != "GoogLeNet":
        raise ValueError(f"Unknown GoogLeNet Architecture: {googlenet_type}")

    # Load the pre-trained version of GoogLeNet
    try:
        weights = models.GoogLeNet_Weights.DEFAULT
        googlenet_model = models.googlenet(weights=weights)
    except RuntimeError as e:
        print(f"{googlenet_type} - Error loading pretrained GoogLeNet model: {e}")
        googlenet_model = models.googlenet(weights=None)

    # Modify the fully connected layer to suit the given number of classes
    googlenet_model.fc = nn.Linear(googlenet_model.fc.in_features, num_classes)

    return googlenet_model
