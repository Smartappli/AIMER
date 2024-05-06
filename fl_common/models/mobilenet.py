import torch.nn as nn
from torchvision import models
from timm import create_model


def get_mobilenet_model(mobilenet_type, num_classes):
    """
    Obtain a MobileNet model with a specified architecture type and modify it for the given number of classes.

    Args:
    - mobilenet_type (str): Type of MobileNet architecture to be loaded.
      Options: 'MobileNet_V2', 'MobileNet_V3_Small', 'MobileNet_V3_Large', 'mobilenetv3_large_075',
               'mobilenetv3_large_100', 'mobilenetv3_small_050', 'mobilenetv3_small_075', 'mobilenetv3_small_100',
               'mobilenetv3_rw', 'tf_mobilenetv3_large_075', 'tf_mobilenetv3_large_100',
               'tf_mobilenetv3_large_minimal_100', 'tf_mobilenetv3_small_075', 'tf_mobilenetv3_small_100',
               'tf_mobilenetv3_small_minimal_100', 'fbnetv3_b', 'fbnetv3_d', 'fbnetv3_g', 'lcnet_035', 'lcnet_050',
               'lcnet_075', 'lcnet_100', 'lcnet_150'.
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
    torch_vision = False
    # Load the pre-trained version of VGG
    if mobilenet_type == 'MobileNet_V2':
        torch_vision = True
        try:
            weights = models.MobileNet_V2_Weights.DEFAULT
            mobilenet_model = models.mobilenet_v2(weights=weights)
        except:
            mobilenet_model = models.mobilenet_v2(weights=None)
    elif mobilenet_type == 'MobileNet_V3_Small':
        torch_vision = True
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            mobilenet_model = models.mobilenet_v3_small(weights=weights)
        except:
            mobilenet_model = models.mobilenet_v3_small(weights=None)
    elif mobilenet_type == 'MobileNet_V3_Large':
        torch_vision = True
        try:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            mobilenet_model = models.mobilenet_v3_large(weights=weights)
        except:
            mobilenet_model = models.mobilenet_v3_large(weights=None)
    elif mobilenet_type == 'mobilenetv3_large_075':
        try:
            mobilenet_model = create_model('mobilenetv3_large_075',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('mobilenetv3_large_075',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'mobilenetv3_large_100':
        try:
            mobilenet_model = create_model('mobilenetv3_large_100',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('mobilenetv3_large_100',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'mobilenetv3_small_050':
        try:
            mobilenet_model = create_model('mobilenetv3_small_050',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('mobilenetv3_small_050',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'mobilenetv3_small_075':
        try:
            mobilenet_model = create_model('mobilenetv3_small_075',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('mobilenetv3_small_075',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'mobilenetv3_small_100':
        try:
            mobilenet_model = create_model('mobilenetv3_small_100',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('mobilenetv3_small_100',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'mobilenetv3_rw':
        try:
            mobilenet_model = create_model('mobilenetv3_rw',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('mobilenetv3_rw',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'tf_mobilenetv3_large_075':
        try:
            mobilenet_model = create_model('tf_mobilenetv3_large_075',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('tf_mobilenetv3_large_075',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'tf_mobilenetv3_large_100':
        try:
            mobilenet_model = create_model('tf_mobilenetv3_large_100',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('tf_mobilenetv3_large_100',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'tf_mobilenetv3_large_minimal_100':
        try:
            mobilenet_model = create_model('tf_mobilenetv3_large_minimal_100',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('tf_mobilenetv3_large_minimal_100',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'tf_mobilenetv3_small_075':
        try:
            mobilenet_model = create_model('tf_mobilenetv3_small_075',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('tf_mobilenetv3_small_075',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'tf_mobilenetv3_small_100':
        try:
            mobilenet_model = create_model('tf_mobilenetv3_small_100',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('tf_mobilenetv3_small_100',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'tf_mobilenetv3_small_minimal_100':
        try:
            mobilenet_model = create_model('tf_mobilenetv3_small_minimal_100',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('tf_mobilenetv3_small_minimal_100',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'fbnetv3_b':
        try:
            mobilenet_model = create_model('fbnetv3_b',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('fbnetv3_b',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'fbnetv3_d':
        try:
            mobilenet_model = create_model('fbnetv3_d',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('fbnetv3_d',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'fbnetv3_g':
        try:
            mobilenet_model = create_model('fbnetv3_g',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('fbnetv3_g',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'lcnet_035':
        try:
            mobilenet_model = create_model('lcnet_035',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('lcnet_035',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'lcnet_050':
        try:
            mobilenet_model = create_model('lcnet_050',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('lcnet_050',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'lcnet_075':
        try:
            mobilenet_model = create_model('lcnet_075',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('lcnet_075',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'lcnet_100':
        try:
            mobilenet_model = create_model('lcnet_100',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('lcnet_100',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mobilenet_type == 'lcnet_150':
        try:
            mobilenet_model = create_model('lcnet_150',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            mobilenet_model = create_model('lcnet_150',
                                           pretrained=False,
                                           num_classes=num_classes)
    else:
        raise ValueError(f'Unknown MobileNet Architecture : {mobilenet_type}')

    if torch_vision:
        # Modify last layer to suit number of classes
        num_features = mobilenet_model.classifier[-1].in_features
        mobilenet_model.classifier[-1] = nn.Linear(num_features, num_classes)

    return mobilenet_model
