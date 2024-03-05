import torch.nn as nn
from torchvision import models

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

    # Load the pre-trained version of RegNet
    if regnet_type == 'RegNet_X_400MF':
        try:
            weights = models.RegNet_X_400MF_Weights.DEFAULT
            regnet_model = models.regnet_x_400mf(weights=weights)
        except:
            regnet_model = models.regnet_x_400mf(weights=None)
    elif regnet_type == 'RegNet_X_800MF':
        try:
            weights = models.RegNet_X_800MF_Weights.DEFAULT
            regnet_model = models.regnet_x_800mf(weights=weights)
        except:
            regnet_model = models.regnet_x_800mf(weights=None)
    elif regnet_type == 'RegNet_X_1_6GF':
        try:
            weights = models.RegNet_X_1_6GF_Weights.DEFAULT
            regnet_model = models.regnet_x_1_6gf(weights=weights)
        except:
            regnet_model = models.regnet_x_1_6gf(weights=None)
    elif regnet_type == 'RegNet_X_3_2GF':
        try:
            weights = models.RegNet_X_3_2GF_Weights.DEFAULT
            regnet_model = models.regnet_x_3_2gf(weights=weights)
        except:
            regnet_model = models.regnet_x_3_2gf(weights=None)
    elif regnet_type == 'RegNet_X_16GF':
        try:
            weights = models.RegNet_X_16GF_Weights.DEFAULT
            regnet_model = models.regnet_x_16gf(weights=weights)
        except:
            regnet_model = models.regnet_x_16gf(weights=None)
    elif regnet_type == 'RegNet_Y_400MF':
        try:
            weights = models.RegNet_Y_400MF_Weights.DEFAULT
            regnet_model = models.regnet_y_400mf(weights=weights)
        except:
            regnet_model = models.regnet_y_400mf(weights=None)
    elif regnet_type == 'RegNet_Y_800MF':
        try:
            weights = models.RegNet_Y_800MF_Weights.DEFAULT
            regnet_model = models.regnet_y_800mf(weights=weights)
        except:
            regnet_model = models.regnet_y_800mf(weights=None)
    elif regnet_type == 'RegNet_Y_1_6GF':
        try:
            weights = models.RegNet_Y_1_6GF_Weights.DEFAULT
            regnet_model = models.regnet_y_1_6gf(weights=weights)
        except:
            regnet_model = models.regnet_y_1_6gf(weights=None)
    elif regnet_type == 'RegNet_Y_3_2GF':
        try:
            weights = models.RegNet_Y_3_2GF_Weights.DEFAULT
            regnet_model = models.regnet_y_3_2gf(weights=weights)
        except:
            regnet_model = models.regnet_y_3_2gf(weights=None)
    elif regnet_type == 'RegNet_Y_16GF':
        try:
            weights = models.RegNet_Y_16GF_Weights.DEFAULT
            regnet_model = models.regnet_y_16gf(weights=weights)
        except:
            regnet_model = models.regnet_y_16gf(weights=None)
    else:
        raise ValueError(f'Unknown RegNet Architecture: {regnet_type}')

    # Modify last layer to suit number of classes
    num_features = regnet_model.fc.in_features
    regnet_model.fc = nn.Linear(num_features, num_classes)

    return regnet_model
