import torch.nn as nn
from torchvision import models

def get_mobilenet_model(mobilenet_type, num_classes):
    """
    Obtain a MobileNet model with a specified architecture type and modify it for the given number of classes.

    Args:
    - mobilenet_type (str): Type of MobileNet architecture to be loaded.
      Options: 'MobileNet_V2', 'MobileNet_V3_Small', 'MobileNet_V3_Large'.
      Default is 'MobileNet_V3_Small'.
    - num_classes (int): Number of output classes for the modified model. Default is 1000.

    Returns:
    - mobilenet_model (torch.nn.Module): The modified MobileNet model.

    Raises:
    - ValueError: If the provided mobilenet_type is not recognized.

    Note:
    - This function loads a pre-trained MobileNet model and modifies its last fully connected layer
      to match the specified number of output classes.

    Example Usage:
    ```python
    # Obtain a MobileNet_V2 model with 10 output classes
    model = get_mobilenet_model(mobilenet_type='MobileNet_V2', num_classes=10)
    ```
    """

    # Load the pre-trained version of VGG
    if mobilenet_type == 'MobileNet_V2':
        try:
            weights = models.MobileNet_V2_Weights.DEFAULT
            mobilenet_model = models.mobilenet_v2(weights=weights)
        except:
            mobilenet_model = models.mobilenet_v2(weights=None)
    elif mobilenet_type == 'MobileNet_V3_Small':
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            mobilenet_model = models.mobilenet_v3_small(weights=weights)
        except:
            mobilenet_model = models.mobilenet_v3_small(weights=None)
    elif mobilenet_type == 'MobileNet_V3_Large':
        try:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            mobilenet_model = models.mobilenet_v3_large(weights=weights)
        except:
            mobilenet_model = models.mobilenet_v3_large(weights=None)
    else:
        raise ValueError(f'Unknown MobileNet Architecture : {mobilenet_type}')

    # Modify last layer to suit number of classes
    num_features = mobilenet_model.classifier[-1].in_features
    mobilenet_model.classifier[-1] = nn.Linear(num_features, num_classes)

    return mobilenet_model
