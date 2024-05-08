import torch.nn as nn
from torchvision import models


def get_resnext_model(resnext_type, num_classes):
    """
    Returns a modified ResNeXt model based on the specified type.

    Parameters:
        - resnext_type (str): Type of ResNeXt architecture.
                             Currently supports 'ResNeXt50_32X4D', 'ResNeXt101_32X8D', and 'ResNeXt101_64X4D'.
        - num_classes (int): Number of classes for the modified last layer.

    Returns:
        - torch.nn.Module: Modified ResNeXt model with the specified number of classes.

    Raises:
        - ValueError: If an unknown ResNeXt architecture is provided.
    """
    # Load the pre-trained version of ResNeXt based on the specified type
    if resnext_type == 'ResNeXt50_32X4D':
        try:
            weights = models.ResNeXt50_32X4D_Weights.DEFAULT
            resnext_model = models.resnext50_32x4d(weights=weights)
        except RuntimeError:
            resnext_model = models.resnext50_32x4d(weights=None)
    elif resnext_type == 'ResNeXt101_32X8D':
        try:
            weights = models.ResNeXt101_32X8D_Weights.DEFAULT
            resnext_model = models.resnext101_32x8d(weights=weights)
        except RuntimeError:
            resnext_model = models.resnext101_32x8d(weights=None)
    elif resnext_type == 'ResNeXt101_64X4D':
        try:
            weights = models.ResNeXt101_64X4D_Weights.DEFAULT
            resnext_model = models.resnext101_64x4d(weights=weights)
        except RuntimeError:
            resnext_model = models.resnext101_64x4d(weights=None)
    else:
        raise ValueError(f'Unknown ResNeXt Architecture: {resnext_type}')

    # Modify the last layer to suit the given number of classes
    num_features = resnext_model.fc.in_features
    resnext_model.fc = nn.Linear(num_features, num_classes)

    return resnext_model
