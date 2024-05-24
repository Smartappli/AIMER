from torch import nn
from torchvision import models
from timm import create_model


def get_regnet_model(regnet_type, num_classes):
    """
    Obtain a RegNet model with a specified architecture type and modify it for the given number of classes.

    Args:
    - regnet_type (str): Type of RegNet architecture to be loaded.
      Options: 'RegNet_X_400MF', 'RegNet_X_800MF', 'RegNet_X_1_6GF', 'RegNet_X_3_2GF', 'RegNet_X_16GF',
               'RegNet_Y_400MF', 'RegNet_Y_800MF', 'RegNet_Y_1_6GF', 'RegNet_Y_3_2GF', 'RegNet_Y_16GF'.
      Default is 'RegNet_X_400MF'.
    - num_classes (int): Number of output classes for the modified model. Default is 1000.

    Returns:
    - regnet_model (torch.nn.Module): The modified RegNet model.

    Raises:
    - ValueError: If the provided regnet_type is not recognized.
    """
    # Mapping of vision types to their corresponding torchvision models and
    # weights
    torchvision_models = {
        "RegNet_X_400MF": (
            models.regnet_x_400mf,
            models.RegNet_X_400MF_Weights,
        ),
        "RegNet_X_800MF": (
            models.regnet_x_800mf,
            models.RegNet_X_800MF_Weights,
        ),
        "RegNet_X_1_6GF": (
            models.regnet_x_1_6gf,
            models.RegNet_X_1_6GF_Weights,
        ),
        "RegNet_X_3_2GF": (
            models.regnet_x_3_2gf,
            models.RegNet_X_3_2GF_Weights,
        ),
        "RegNet_X_16GF": (models.regnet_x_16gf, models.RegNet_X_16GF_Weights),
        "RegNet_Y_400MF": (
            models.regnet_y_400mf,
            models.RegNet_Y_400MF_Weights,
        ),
        "RegNet_Y_800MF": (
            models.regnet_y_800mf,
            models.RegNet_Y_800MF_Weights,
        ),
        "RegNet_Y_1_6GF": (
            models.regnet_y_1_6gf,
            models.RegNet_Y_1_6GF_Weights,
        ),
        "RegNet_Y_3_2GF": (
            models.regnet_y_3_2gf,
            models.RegNet_Y_3_2GF_Weights,
        ),
        "RegNet_Y_16GF": (models.regnet_y_16gf, models.RegNet_Y_16GF_Weights),
    }

    timm_models = [
        "regnetx_002",
        "regnetx_004",
        "regnetx_004_tv",
        "regnetx_006",
        "regnetx_008",
        "regnetx_016",
        "regnetx_032",
        "regnetx_040",
        "regnetx_064",
        "regnetx_080",
        "regnetx_120",
        "regnetx_160",
        "regnetx_320",
        "regnety_002",
        "regnety_004",
        "regnety_006",
        "regnety_008",
        "regnety_008_tv",
        "regnety_016",
        "regnety_032",
        "regnety_040",
        "regnety_064",
        "regnety_080",
        "regnety_080_tv",
        "regnety_120",
        "regnety_160",
        "regnety_320",
        "regnety_640",
        "regnety_1280",
        "regnety_2560",
        "regnety_040_sgn",
        "regnetv_040",
        "regnetv_064",
        "regnetz_005",
        "regnetz_040",
        "regnetz_040_h",
    ]

    # Check if the vision type is from torchvision
    if regnet_type in torchvision_models:
        model_func, weights_class = torchvision_models[regnet_type]
        try:
            weights = weights_class.DEFAULT
            regnet_model = model_func(weights=weights)
        except RuntimeError as e:
            print(f"{regnet_type} - Error loading pretrained model: {e}")
            regnet_model = model_func(weights=None)

        # Modify last layer to suit number of classes
        num_features = regnet_model.fc.in_features
        regnet_model.fc = nn.Linear(num_features, num_classes)

    # Check if the vision type is from the 'timm' library
    elif regnet_type in timm_models:
        try:
            regnet_model = create_model(
                regnet_type, pretrained=True, num_classes=num_classes
            )
        except RuntimeError as e:
            print(f"{regnet_type} - Error loading pretrained model: {e}")
            regnet_model = create_model(
                regnet_type, pretrained=False, num_classes=num_classes
            )
    else:
        raise ValueError(f"Unknown RegNet Architecture: {regnet_type}")

    return regnet_model
