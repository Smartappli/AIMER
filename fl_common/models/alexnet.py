import torch.nn as nn
from torchvision import models


def get_alexnet_model(alexnet_type, num_classes):
    """
    Load a pre-trained AlexNet model of the specified type and modify its
    last layer to accommodate the given number of classes.

    Parameters:
    - alexnet_type (str): Type of AlexNet architecture, supported type:
        - 'AlexNet'
    - num_classes (int): Number of output classes for the modified last layer.

    Returns:
    - torch.nn.Module: Modified AlexNet model with the specified architecture
      and last layer adapted for the given number of classes.

    Raises:
    - ValueError: If an unknown AlexNet architecture type is provided.
    """
    # Load the pre-trained version of AlexNet based on the specified type
    if alexnet_type == 'AlexNet':
        try:
            weights = models.AlexNet_Weights.DEFAULT
            alexnet_model = models.alexnet(weights=weights)
        except OSError:
            alexnet_model = models.alexnet(weights=None)
    else:
        raise ValueError(f'Unknown AlexNet Architecture: {alexnet_type}')

    # Modify the last layer to suit the given number of classes
    num_features = alexnet_model.classifier[6].in_features
    alexnet_model.classifier[6] = nn.Linear(num_features, num_classes)

    return alexnet_model
