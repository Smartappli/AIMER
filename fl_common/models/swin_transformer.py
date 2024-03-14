import torch.nn as nn
from torchvision import models


def get_swin_model(swin_type, num_classes):
    """
    Loads a pre-trained Swin Transformer model based on the specified Swin type
    and modifies the last layer for a given number of output classes.

    Parameters:
    - swin_type (str, optional): Type of Swin Transformer model. Default is 'Swin_T'.
    - num_classes (int, optional): Number of output classes. Default is 1000.

    Returns:
    - swin_model (torch.nn.Module): Modified Swin Transformer model with the last layer
      adjusted for the specified number of output classes.

    Raises:
    - ValueError: If the specified Swin type is unknown or if the model does not have
      a known structure with a linear last layer.
    """

    # Load the pre-trained version of DenseNet
    if swin_type == 'Swin_T':
        try:
            weights = models.Swin_T_Weights.DEFAULT
            swin_model = models.swin_t(weights=weights)
        except:
            swin_model = models.swin_t(weights=None)
    elif swin_type == 'Swin_S':
        try:
            weights = models.Swin_S_Weights.DEFAULT
            swin_model = models.swin_s(weights=weights)
        except:
            swin_model = models.swin_s(weights=None)
    elif swin_type == 'Swin_B':
        try:
            weights = models.Swin_B_Weights.DEFAULT
            swin_model = models.swin_b(weights=weights)
        except:
            swin_model = models.swin_b(weights=None)
    elif swin_type == 'Swin_V2_T':
        try:
            weights = models.Swin_V2_T_Weights.DEFAULT
            swin_model = models.swin_v2_t(weights=weights)
        except:
            swin_model = models.swin_v2_t(weights=None)
    elif swin_type == 'Swin_V2_S':
        try:
            weights = models.Swin_V2_S_Weights.DEFAULT
            swin_model = models.swin_v2_s(weights=weights)
        except:
            swin_model = models.swin_v2_s(weights=None)
    elif swin_type == 'Swin_V2_B':
        try:
            weights = models.Swin_V2_B_Weights.DEFAULT
            swin_model = models.swin_v2_b(weights=weights)
        except:
            swin_model = models.swin_v2_b(weights=None)
    else:
        raise ValueError(f'Unknown DenseNet Architecture : {swin_type}')

    # Modify last layer to suit number of classes
    if hasattr(swin_model, 'head') and isinstance(swin_model.head, nn.Linear):
        num_features = swin_model.head.in_features
        swin_model.head = nn.Linear(num_features, num_classes)
    else:
        raise ValueError('Model does not have a known structure.')

    return swin_model
