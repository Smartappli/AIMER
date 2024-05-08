import torch.nn as nn
from torchvision import models


def get_shufflenet_model(shufflenet_type, num_classes):
    """
    Load a pre-trained ShuffleNet model of the specified type and modify its
    last layer to accommodate the given number of classes.

    Parameters:
    - shufflenet_type (str): Type of ShuffleNet architecture, supported types:
        - 'ShuffleNet_V2_X0_5'
        - 'ShuffleNet_V2_X1_0'
        - 'ShuffleNet_V2_X1_5'
        - 'ShuffleNet_V2_X2_0'
    - num_classes (int): Number of output classes for the modified last layer.

    Returns:
    - torch.nn.Module: Modified ShuffleNet model with the specified architecture
      and last layer adapted for the given number of classes.

    Raises:
    - ValueError: If an unknown ShuffleNet architecture type is provided.
    """

    # Load the pre-trained version of DenseNet
    if shufflenet_type == 'ShuffleNet_V2_X0_5':
        try:
            weights = models.ShuffleNet_V2_X0_5_Weights.DEFAULT
            shufflenet_model = models.shufflenet_v2_x0_5(weights=weights)
        except OSError:
            shufflenet_model = models.shufflenet_v2_x0_5(weights=None)
    elif shufflenet_type == 'ShuffleNet_V2_X1_0':
        try:
            weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT
            shufflenet_model = models.shufflenet_v2_x1_0(weights=weights)
        except OSError:
            shufflenet_model = models.shufflenet_v2_x1_0(weights=None)
    elif shufflenet_type == 'ShuffleNet_V2_X1_5':
        try:
            weights = models.ShuffleNet_V2_X1_5_Weights.DEFAULT
            shufflenet_model = models.shufflenet_v2_x1_5(weights=weights)
        except OSError:
            shufflenet_model = models.shufflenet_v2_x1_5(weights=None)
    elif shufflenet_type == 'ShuffleNet_V2_X2_0':
        try:
            weights = models.ShuffleNet_V2_X2_0_Weights.DEFAULT
            shufflenet_model = models.shufflenet_v2_x2_0(weights=weights)
        except OSError:
            shufflenet_model = models.shufflenet_v2_x2_0(weights=None)
    else:
        raise ValueError(f'Unknown ShuffleNet Architecture: {shufflenet_type}')

    # Modify last layer to suit number of classes
    num_features = shufflenet_model.fc.in_features
    shufflenet_model.fc = nn.Linear(num_features, num_classes)

    return shufflenet_model
