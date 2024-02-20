import torch.nn as nn
from torchvision import models


def get_convnext_model(convnext_type, num_classes):
    """
    Obtain a ConvNeXt model with a specified architecture type and modify it for the given number of classes.

    Args:
    - convnext_type (str): Type of ConvNeXt architecture to be loaded. Options: 'ConvNeXt_Tiny', 'ConvNeXt_Small',
      'ConvNeXt_Base', 'ConvNeXt_Large'. Default is 'ConvNeXt_Large'.
    - num_classes (int): Number of output classes for the modified model. Default is 1000.

    Returns:
    - convnext_model (torch.nn.Module): The modified ConvNeXt model.

    Raises:
    - ValueError: If the provided convnext_type is not recognized or the model structure is not as expected.

    Note:
    - This function loads a pre-trained ConvNeXt model and modifies its last fully connected layer to
      match the specified number of output classes.

    Example Usage:
    ```python
    # Obtain a ConvNeXt model with 10 output classes
    model = get_convnext_model(convnext_type='ConvNeXt_Small', num_classes=10)
    ```
    """

    # Load the pre-trained version of DenseNet
    if convnext_type == 'ConvNeXt_Tiny':
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        convnext_model = models.convnext_tiny(weights=weights)
    elif convnext_type == 'ConvNeXt_Small':
        weights = models.ConvNeXt_Small_Weights.DEFAULT
        convnext_model = models.convnext_small(weights=weights)
    elif convnext_type == 'ConvNeXt_Base':
        weights = models.ConvNeXt_Base_Weights.DEFAULT
        convnext_model = models.convnext_base(weights=weights)
    elif convnext_type == 'ConvNeXt_Large':
        weights = models.ConvNeXt_Large_Weights.DEFAULT
        convnext_model = models.convnext_large(weights=weights)
    else:
        raise ValueError(f'Unknown ConvNeXt Architecture : {convnext_type}')

    # Modify last layer to suit number of classes
    for layer in reversed(convnext_model.classifier):
        if isinstance(layer, nn.Linear):
            # num_features = layer.in_features
            layer.out_features = num_classes
            break

    return convnext_model
