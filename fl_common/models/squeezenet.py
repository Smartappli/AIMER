import torch.nn as nn
from torchvision import models


def get_squeezenet_model(squeezenet_type, num_classes):
    """
    Returns a modified SqueezeNet model based on the specified type.

    Parameters:
        - squeezenet_type (str): Type of SqueezeNet architecture.
                                Currently supports 'SqueezeNet1_0' and 'SqueezeNet1_1'.
        - num_classes (int): Number of classes for the modified last layer.

    Returns:
        - torch.nn.Module: Modified SqueezeNet model with the specified number of classes.

    Raises:
        - ValueError: If an unknown SqueezeNet architecture is provided.
    """
    # Load the pre-trained version of SqueezeNet based on the specified type
    if squeezenet_type == 'SqueezeNet1_0':
        try:
            weights = models.SqueezeNet1_0_Weights.DEFAULT
            squeezenet_model = models.squeezenet1_0(weights=weights)
        except RuntimeError:
            squeezenet_model = models.squeezenet1_0(weights=None)
    elif squeezenet_type == 'SqueezeNet1_1':
        try:
            weights = models.SqueezeNet1_1_Weights.DEFAULT
            squeezenet_model = models.squeezenet1_1(weights=weights)
        except RuntimeError:
            squeezenet_model = models.squeezenet1_1(weights=None)
    else:
        raise ValueError(f'Unknown SqueezeNet Architecture: {squeezenet_type}')

    # Modify the last layer to suit the given number of classes
    num_features = squeezenet_model.classifier[1].in_channels
    squeezenet_model.classifier[1] = nn.Conv2d(num_features, num_classes, kernel_size=(1, 1), stride=(1, 1))

    return squeezenet_model
