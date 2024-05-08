import torch.nn as nn
from torchvision import models


def get_wide_resnet_model(wide_resnet_type, num_classes):
    """
    Returns a modified Wide ResNet model based on the specified type.

    Parameters:
        - wide_resnet_type (str): Type of Wide ResNet architecture.
                                 Currently supports 'Wide_ResNet50_2' and 'Wide_ResNet101_2'.
        - num_classes (int): Number of classes for the modified last layer.

    Returns:
        - torch.nn.Module: Modified Wide ResNet model with the specified number of classes.

    Raises:
        - ValueError: If an unknown Wide ResNet architecture is provided.
    """
    # Load the pre-trained version of Wide ResNet based on the specified type
    if wide_resnet_type == 'Wide_ResNet50_2':
        try:
            weights = models.Wide_ResNet50_2_Weights.DEFAULT
            wide_resnet_model = models.wide_resnet50_2(weights=weights)
        except OSError:
            wide_resnet_model = models.wide_resnet50_2(weights=None)
    elif wide_resnet_type == 'Wide_ResNet101_2':
        try:
            weights = models.Wide_ResNet101_2_Weights.DEFAULT
            wide_resnet_model = models.wide_resnet101_2(weights=weights)
        except OSError:
            wide_resnet_model = models.wide_resnet101_2(weights=None)
    else:
        raise ValueError(f'Unknown Wide ResNet Architecture: {wide_resnet_type}')

    # Modify the last layer to suit the given number of classes
    num_features = wide_resnet_model.fc.in_features
    wide_resnet_model.fc = nn.Linear(num_features, num_classes)

    return wide_resnet_model
