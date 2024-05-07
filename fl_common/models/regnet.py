import torch.nn as nn
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

    Note:
    - This function loads a pre-trained RegNet model and modifies its last fully connected layer
      to match the specified number of output classes.

    Example Usage:
    ```python
    # Obtain a RegNet_X_400MF model with 10 output classes
    model = get_regnet_model(regnet_type='RegNet_X_400MF', num_classes=10)
    ```
    """
    torch_vision = False
    # Load the pre-trained version of RegNet
    if regnet_type == 'RegNet_X_400MF':
        torch_vision = True
        try:
            weights = models.RegNet_X_400MF_Weights.DEFAULT
            regnet_model = models.regnet_x_400mf(weights=weights)
        except ValueError:
            regnet_model = models.regnet_x_400mf(weights=None)
    elif regnet_type == 'RegNet_X_800MF':
        torch_vision = True
        try:
            weights = models.RegNet_X_800MF_Weights.DEFAULT
            regnet_model = models.regnet_x_800mf(weights=weights)
        except ValueError:
            regnet_model = models.regnet_x_800mf(weights=None)
    elif regnet_type == 'RegNet_X_1_6GF':
        torch_vision = True
        try:
            weights = models.RegNet_X_1_6GF_Weights.DEFAULT
            regnet_model = models.regnet_x_1_6gf(weights=weights)
        except ValueError:
            regnet_model = models.regnet_x_1_6gf(weights=None)
    elif regnet_type == 'RegNet_X_3_2GF':
        torch_vision = True
        try:
            weights = models.RegNet_X_3_2GF_Weights.DEFAULT
            regnet_model = models.regnet_x_3_2gf(weights=weights)
        except ValueError:
            regnet_model = models.regnet_x_3_2gf(weights=None)
    elif regnet_type == 'RegNet_X_16GF':
        torch_vision = True
        try:
            weights = models.RegNet_X_16GF_Weights.DEFAULT
            regnet_model = models.regnet_x_16gf(weights=weights)
        except ValueError:
            regnet_model = models.regnet_x_16gf(weights=None)
    elif regnet_type == 'RegNet_Y_400MF':
        torch_vision = True
        try:
            weights = models.RegNet_Y_400MF_Weights.DEFAULT
            regnet_model = models.regnet_y_400mf(weights=weights)
        except ValueError:
            regnet_model = models.regnet_y_400mf(weights=None)
    elif regnet_type == 'RegNet_Y_800MF':
        torch_vision = True
        try:
            weights = models.RegNet_Y_800MF_Weights.DEFAULT
            regnet_model = models.regnet_y_800mf(weights=weights)
        except ValueError:
            regnet_model = models.regnet_y_800mf(weights=None)
    elif regnet_type == 'RegNet_Y_1_6GF':
        torch_vision = True
        try:
            weights = models.RegNet_Y_1_6GF_Weights.DEFAULT
            regnet_model = models.regnet_y_1_6gf(weights=weights)
        except ValueError:
            regnet_model = models.regnet_y_1_6gf(weights=None)
    elif regnet_type == 'RegNet_Y_3_2GF':
        torch_vision = True
        try:
            weights = models.RegNet_Y_3_2GF_Weights.DEFAULT
            regnet_model = models.regnet_y_3_2gf(weights=weights)
        except ValueError:
            regnet_model = models.regnet_y_3_2gf(weights=None)
    elif regnet_type == 'RegNet_Y_16GF':
        torch_vision = True
        try:
            weights = models.RegNet_Y_16GF_Weights.DEFAULT
            regnet_model = models.regnet_y_16gf(weights=weights)
        except ValueError:
            regnet_model = models.regnet_y_16gf(weights=None)
    elif regnet_type == 'regnetx_002':
        try:
            regnet_model = create_model('regnetx_002',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetx_002',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnetx_004':
        try:
            regnet_model = create_model('regnetx_004',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetx_004',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnetx_006':
        try:
            regnet_model = create_model('regnetx_006',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetx_006',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnetx_008':
        try:
            regnet_model = create_model('regnetx_008',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetx_008',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnetx_016':
        try:
            regnet_model = create_model('regnetx_016',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetx_016',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnetx_032':
        try:
            regnet_model = create_model('regnetx_032',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetx_032',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnetx_040':
        try:
            regnet_model = create_model('regnetx_040',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetx_040',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnetx_064':
        try:
            regnet_model = create_model('regnetx_064',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetx_064',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnetx_080':
        try:
            regnet_model = create_model('regnetx_080',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetx_080',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnetx_120':
        try:
            regnet_model = create_model('regnetx_120',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetx_120',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnetx_160':
        try:
            regnet_model = create_model('regnetx_160',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetx_160',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnetx_320':
        try:
            regnet_model = create_model('regnetx_320',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetx_320',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_002':
        try:
            regnet_model = create_model('regnety_002',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_002',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_004':
        try:
            regnet_model = create_model('regnety_004',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_004',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_006':
        try:
            regnet_model = create_model('regnety_006',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_006',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_008':
        try:
            regnet_model = create_model('regnety_008',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_008',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_008_tv':
        try:
            regnet_model = create_model('regnety_008_tv',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_008_tv',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_016':
        try:
            regnet_model = create_model('regnety_016',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_016',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_032':
        try:
            regnet_model = create_model('regnety_032',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_032',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_040':
        try:
            regnet_model = create_model('regnety_040',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_040',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_064':
        try:
            regnet_model = create_model('regnety_064',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_064',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_080':
        try:
            regnet_model = create_model('regnety_080',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_080',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_080_tv':
        try:
            regnet_model = create_model('regnety_080_tv',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_080_tv',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_120':
        try:
            regnet_model = create_model('regnety_120',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_120',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_160':
        try:
            regnet_model = create_model('regnety_160',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_160',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_320':
        try:
            regnet_model = create_model('regnety_320',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_320',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_640':
        try:
            regnet_model = create_model('regnety_640',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_640',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_1280':
        try:
            regnet_model = create_model('regnety_1280',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_1280',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_2560':
        try:
            regnet_model = create_model('regnety_2560',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_2560',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnety_040_sgn':
        try:
            regnet_model = create_model('regnety_040_sgn',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnety_040_sgn',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnetv_040':
        try:
            regnet_model = create_model('regnetv_040',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetv_040',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnetv_064':
        try:
            regnet_model = create_model('regnetv_064',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetv_064',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnetz_005':
        try:
            regnet_model = create_model('regnetz_005',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetz_005',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnetz_040':
        try:
            regnet_model = create_model('regnetz_040',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetz_040',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif regnet_type == 'regnetz_040_h':
        try:
            regnet_model = create_model('regnetz_040_h',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            regnet_model = create_model('regnetz_040_h',
                                        pretrained=False,
                                        num_classes=num_classes)
    else:
        raise ValueError(f'Unknown RegNet Architecture: {regnet_type}')

    if torch_vision:
        # Modify last layer to suit number of classes
        num_features = regnet_model.fc.in_features
        regnet_model.fc = nn.Linear(num_features, num_classes)

    return regnet_model
