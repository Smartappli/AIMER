import torch.nn as nn
from torchvision import models

def get_vgg_model(vgg_type, num_classes):
    """
    Obtain a VGG model with a specified architecture type and modify it for the given number of classes.

    Args:
    - vgg_type (str): Type of VGG architecture to be loaded.
      Options: 'VGG11', 'VGG11_BN', 'VGG13', 'VGG13_BN', 'VGG16', 'VGG16_BN', 'VGG19', 'VGG19_BN'.
      Default is 'VGG16'.
    - num_classes (int): Number of output classes for the modified model. Default is 1000.

    Returns:
    - vgg_model (torch.nn.Module): The modified VGG model.

    Raises:
    - ValueError: If the provided vgg_type is not recognized.

    Note:
    - This function loads a pre-trained VGG model and modifies its last fully connected layer
      to match the specified number of output classes.

    Example Usage:
    ```python
    # Obtain a VGG16 model with 10 output classes
    model = get_vgg_model(vgg_type='VGG16', num_classes=10)
    ```
    """

    # Load the pre-trained version of VGG
    if vgg_type == 'VGG11':
        weights = models.VGG11_Weights.DEFAULT
        vgg_model = models.vgg11(weights=weights)
    elif vgg_type == 'VGG11_BN':
        weights = models.VGG11_BN_Weights.DEFAULT
        vgg_model = models.vgg11_bn(weights=weights)
    elif vgg_type == 'VGG13':
        weights = models.VGG13_Weights.DEFAULT
        vgg_model = models.vgg13(weights=weights)
    elif vgg_type == 'VGG13_BN':
        weights = models.VGG13_BN_Weights.DEFAULT
        vgg_model = models.vgg13_bn(weights=weights)
    elif vgg_type == 'VGG16':
        weights = models.VGG16_Weights.DEFAULT
        vgg_model = models.vgg16(weights=weights)
    elif vgg_type == 'VGG16_BN':
        weights = models.VGG16_BN_Weights.DEFAULT
        vgg_model = models.vgg16_bn(weights=weights)
    elif vgg_type == 'VGG19':
        weights = models.VGG19_Weights.DEFAULT
        vgg_model = models.vgg19(weights=weights)
    elif vgg_type == 'VGG19_BN':
        weights = models.VGG19_BN_Weights.DEFAULT
        vgg_model = models.vgg19_bn(weights=weights)
    else:
        raise ValueError(f'Unknown VGG Architecture : {vgg_type}')

    # Modify last layer to suit number of classes
    num_features = vgg_model.classifier[-1].in_features
    vgg_model.classifier[-1] = nn.Linear(num_features, num_classes)

    return vgg_model
