import torch.nn as nn
from torchvision import models

def get_maxvit_model(maxvit_type, num_classes):
    """
    Returns a modified MaxVit model based on the specified type.

    Parameters:
        - maxvit_type (str): Type of MaxVit architecture.
                            Currently supports 'MaxVit_T'.
        - num_classes (int): Number of classes for the modified last layer.

    Returns:
        - torch.nn.Module: Modified MaxVit model with the specified number of classes.

    Raises:
        - ValueError: If an unknown MaxVit architecture type is provided.
    """
    # Load the pre-trained version of MaxVit based on the specified type
    if maxvit_type == 'MaxVit_T':
        try:
            weights = models.MaxVit_T_Weights.DEFAULT
            maxvit_model = models.maxvit_t(weights=weights)
        except:
            maxvit_model = models.maxvit_t(weights=None)
    else:
        raise ValueError(f'Unknown MaxVit Architecture: {maxvit_type}')

    # Modify the last layer to suit the given number of classes
    num_features = maxvit_model.classifier[-1].in_features
    maxvit_model.classifier[-1] = nn.Linear(num_features, num_classes)

    return maxvit_model
