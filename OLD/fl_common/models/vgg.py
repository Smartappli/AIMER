from timm import create_model
from torch import nn
from torchvision import models


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
    """
    torchvision_models = {
        "VGG11": (models.vgg11, models.VGG11_Weights),
        "VGG11_BN": (models.vgg11_bn, models.VGG11_BN_Weights),
        "VGG13": (models.vgg13, models.VGG13_Weights),
        "VGG13_BN": (models.vgg13_bn, models.VGG13_BN_Weights),
        "VGG16": (models.vgg16, models.VGG16_Weights),
        "VGG16_BN": (models.vgg16_bn, models.VGG16_BN_Weights),
        "VGG19": (models.vgg19, models.VGG19_Weights),
        "VGG19_BN": (models.vgg19_bn, models.VGG19_BN_Weights),
    }

    timm_models = [
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
    ]

    # Check if the vision type is from torchvision
    if vgg_type in torchvision_models:
        model_func, weights_class = torchvision_models[vgg_type]
        try:
            weights = weights_class.DEFAULT
            vgg_model = model_func(weights=weights)
        except RuntimeError as e:
            print(f"{vgg_type} - Error loading pretrained model: {e}")
            vgg_model = model_func(weights=None)

        # Modify last layer to suit number of classes
        num_features = vgg_model.classifier[-1].in_features
        vgg_model.classifier[-1] = nn.Linear(num_features, num_classes)

    # Check if the vision type is from the 'timm' library
    elif vgg_type in timm_models:
        try:
            vgg_model = create_model(
                vgg_type,
                pretrained=True,
                num_classes=num_classes,
            )
        except RuntimeError as e:
            print(f"{vgg_type} - Error loading pretrained model: {e}")
            vgg_model = create_model(
                vgg_type,
                pretrained=False,
                num_classes=num_classes,
            )
    else:
        msg = f"Unknown VGG Architecture : {vgg_type}"
        raise ValueError(msg)

    return vgg_model
