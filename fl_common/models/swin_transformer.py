import torch.nn as nn
from torchvision import models
from timm import create_model


def get_swin_transformer_model(swin_type, num_classes):
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

    torch_vision = False
    # Load the pre-trained version of DenseNet
    if swin_type == 'Swin_T':
        torch_vision = True
        try:
            weights = models.Swin_T_Weights.DEFAULT
            swin_model = models.swin_t(weights=weights)
        except RuntimeError:
            swin_model = models.swin_t(weights=None)
    elif swin_type == 'Swin_S':
        torch_vision = True
        try:
            weights = models.Swin_S_Weights.DEFAULT
            swin_model = models.swin_s(weights=weights)
        except RuntimeError:
            swin_model = models.swin_s(weights=None)
    elif swin_type == 'Swin_B':
        torch_vision = True
        try:
            weights = models.Swin_B_Weights.DEFAULT
            swin_model = models.swin_b(weights=weights)
        except RuntimeError:
            swin_model = models.swin_b(weights=None)
    elif swin_type == 'Swin_V2_T':
        torch_vision = True
        try:
            weights = models.Swin_V2_T_Weights.DEFAULT
            swin_model = models.swin_v2_t(weights=weights)
        except RuntimeError:
            swin_model = models.swin_v2_t(weights=None)
    elif swin_type == 'Swin_V2_S':
        torch_vision = True
        try:
            weights = models.Swin_V2_S_Weights.DEFAULT
            swin_model = models.swin_v2_s(weights=weights)
        except RuntimeError:
            swin_model = models.swin_v2_s(weights=None)
    elif swin_type == 'Swin_V2_B':
        torch_vision = True
        try:
            weights = models.Swin_V2_B_Weights.DEFAULT
            swin_model = models.swin_v2_b(weights=weights)
        except RuntimeError:
            swin_model = models.swin_v2_b(weights=None)
    elif swin_type == "swin_tiny_patch4_window7_224":
        try:
            swin_model = create_model('swin_tiny_patch4_window7_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except RuntimeError:
            swin_model = create_model('swin_tiny_patch4_window7_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swin_small_patch4_window7_224":
        try:
            swin_model = create_model('swin_small_patch4_window7_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except RuntimeError:
            swin_model = create_model('swin_small_patch4_window7_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swin_base_patch4_window7_224":
        try:
            swin_model = create_model('swin_base_patch4_window7_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except RuntimeError:
            swin_model = create_model('swin_base_patch4_window7_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swin_base_patch4_window12_384":
        try:
            swin_model = create_model('swin_base_patch4_window12_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except RuntimeError:
            swin_model = create_model('swin_base_patch4_window12_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swin_large_patch4_window7_224":
        try:
            swin_model = create_model('swin_large_patch4_window7_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except RuntimeError:
            swin_model = create_model('swin_large_patch4_window7_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swin_large_patch4_window12_384":
        try:
            swin_model = create_model('swin_large_patch4_window12_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except RuntimeError:
            swin_model = create_model('swin_large_patch4_window12_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swin_s3_tiny_224":
        try:
            swin_model = create_model('swin_s3_tiny_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except RuntimeError:
            swin_model = create_model('swin_s3_tiny_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swin_s3_small_224":
        try:
            swin_model = create_model('swin_s3_small_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except RuntimeError:
            swin_model = create_model('swin_s3_small_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swin_s3_base_224":
        try:
            swin_model = create_model('swin_s3_base_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except RuntimeError:
            swin_model = create_model('swin_s3_base_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Swin Transformer Architecture : {swin_type}')

    # Modify last layer to suit number of classes
    if torch_vision:
        if hasattr(swin_model, 'head') and isinstance(swin_model.head, nn.Linear):
            num_features = swin_model.head.in_features
            swin_model.head = nn.Linear(num_features, num_classes)
        else:
            raise ValueError('Model does not have a known structure.')

    return swin_model
