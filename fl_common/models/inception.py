import torch.nn as nn
from torchvision import models


def get_inception_model(inception_type, num_classes):
    """
    Load a pre-trained Inception model of the specified type and modify its
    last layer to accommodate the given number of classes.

    Parameters:
    - inception_type (str): Type of Inception architecture, supported type:
        - 'Inception_V3'
    - num_classes (int): Number of output classes for the modified last layer.

    Returns:
    - torch.nn.Module: Modified Inception model with the specified architecture
      and last layer adapted for the given number of classes.

    Raises:
    - ValueError: If an unknown Inception architecture type is provided.
    """
    # Load the pre-trained version of Inception based on the specified type
    if inception_type == 'Inception_V3':
        weights = models.Inception_V3_Weights.DEFAULT
        inception_model = models.inception_v3(weights=weights)
    else:
        raise ValueError(f'Unknown Inception Architecture: {inception_type}')

    # Modify the last layer to suit the given number of classes
    num_features = inception_model.fc.in_features
    inception_model.fc = nn.Linear(num_features, num_classes)

    return inception_model