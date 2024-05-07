import torch.nn as nn
from torchvision import models
from timm import create_model


def get_vgg_model(vgg_type, num_classes):
    """
    Obtain a VGG model with a specified architecture type and modify it for the given number of classes.

    Args:
    - vgg_type (str): Type of VGG architecture to be loaded.
      Options: 'VGG11', 'VGG11_BN', 'VGG13', 'VGG13_BN', 'VGG16', 'VGG16_BN', 'VGG19', 'VGG19_BN',
               'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'.
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
    torch_vision = False
    # Load the pre-trained version of VGG
    if vgg_type == 'VGG11':
        torch_vision = True
        try:
            weights = models.VGG11_Weights.DEFAULT
            vgg_model = models.vgg11(weights=weights)
        except Exception:
            vgg_model = models.vgg11(weights=None)
    elif vgg_type == 'VGG11_BN':
        torch_vision = True
        try:
            weights = models.VGG11_BN_Weights.DEFAULT
            vgg_model = models.vgg11_bn(weights=weights)
        except Exception:
            vgg_model = models.vgg11_bn(weights=None)
    elif vgg_type == 'VGG13':
        torch_vision = True
        try:
            weights = models.VGG13_Weights.DEFAULT
            vgg_model = models.vgg13(weights=weights)
        except Exception:
            vgg_model = models.vgg13(weights=None)
    elif vgg_type == 'VGG13_BN':
        torch_vision = True
        try:
            weights = models.VGG13_BN_Weights.DEFAULT
            vgg_model = models.vgg13_bn(weights=weights)
        except Exception:
            vgg_model = models.vgg13_bn(weights=None)
    elif vgg_type == 'VGG16':
        torch_vision = True
        try:
            weights = models.VGG16_Weights.DEFAULT
            vgg_model = models.vgg16(weights=weights)
        except Exception:
            vgg_model = models.vgg16(weights=None)
    elif vgg_type == 'VGG16_BN':
        torch_vision = True
        try:
            weights = models.VGG16_BN_Weights.DEFAULT
            vgg_model = models.vgg16_bn(weights=weights)
        except Exception:
            vgg_model = models.vgg16_bn(weights=None)
    elif vgg_type == 'VGG19':
        torch_vision = True
        try:
            weights = models.VGG19_Weights.DEFAULT
            vgg_model = models.vgg19(weights=weights)
        except Exception:
            vgg_model = models.vgg19(weights=None)
    elif vgg_type == 'VGG19_BN':
        torch_vision = True
        try:
            weights = models.VGG19_BN_Weights.DEFAULT
            vgg_model = models.vgg19_bn(weights=weights)
        except Exception:
            vgg_model = models.vgg19_bn(weights=None)
    elif vgg_type == "vgg11":
        try:
            vgg_model = create_model('vgg11',
                                     pretrained=True,
                                     num_classes=num_classes)
        except Exception:
            vgg_model = create_model('vgg11',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif vgg_type == "vgg11_bn":
        try:
            vgg_model = create_model('vgg11_bn',
                                     pretrained=True,
                                     num_classes=num_classes)
        except Exception:
            vgg_model = create_model('vgg11_bn',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif vgg_type == "vgg13":
        try:
            vgg_model = create_model('vgg13',
                                     pretrained=True,
                                     num_classes=num_classes)
        except Exception:
            vgg_model = create_model('vgg13',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif vgg_type == "vgg13_bn":
        try:
            vgg_model = create_model('vgg13_bn',
                                     pretrained=True,
                                     num_classes=num_classes)
        except Exception:
            vgg_model = create_model('vgg13_bn',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif vgg_type == "vgg16":
        try:
            vgg_model = create_model('vgg16',
                                     pretrained=True,
                                     num_classes=num_classes)
        except Exception:
            vgg_model = create_model('vgg16',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif vgg_type == "vgg16_bn":
        try:
            vgg_model = create_model('vgg16_bn',
                                     pretrained=True,
                                     num_classes=num_classes)
        except Exception:
            vgg_model = create_model('vgg16_bn',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif vgg_type == "vgg19":
        try:
            vgg_model = create_model('vgg19',
                                     pretrained=True,
                                     num_classes=num_classes)
        except Exception:
            vgg_model = create_model('vgg19',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif vgg_type == "vgg19_bn":
        try:
            vgg_model = create_model('vgg19_bn',
                                     pretrained=True,
                                     num_classes=num_classes)
        except Exception:
            vgg_model = create_model('vgg19_bn',
                                     pretrained=False,
                                     num_classes=num_classes)
    else:
        raise ValueError(f'Unknown VGG Architecture : {vgg_type}')

    if torch_vision:
        # Modify last layer to suit number of classes
        num_features = vgg_model.classifier[-1].in_features
        vgg_model.classifier[-1] = nn.Linear(num_features, num_classes)

    return vgg_model
