import torch.nn as nn
from torchvision import models
from timm import create_model


def get_inception_model(inception_type, num_classes):
    """
    Load a pre-trained Inception model of the specified type and modify its
    last layer to accommodate the given number of classes.

    Parameters:
    - inception_type (str): Type of Inception architecture, supported types:
        - 'Inception_V3'
        - 'inception_v4'
        - 'inception_resnet_v2'
    - num_classes (int): Number of output classes for the modified last layer.

    Returns:
    - torch.nn.Module: Modified Inception model with the specified architecture
      and last layer adapted for the given number of classes.

    Raises:
    - ValueError: If an unknown Inception architecture type is provided.
    """
    # Load the pre-trained version of Inception based on the specified type
    if inception_type == 'Inception_V3':
        weights = models.Inception_V3_Weights.DEFAULT
        inception_model = models.inception_v3(weights=weights)

        # Modify the last layer to suit the given number of classes
        num_features = inception_model.fc.in_features
        inception_model.fc = nn.Linear(num_features, num_classes)
    elif inception_type == 'inception_v4':
        try:
            inception_model = create_model('inception_v4',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            inception_model = create_model('inception_v4',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif inception_type == 'inception_resnet_v2':
        try:
            inception_model = create_model('inception_resnet_v2',
                                           pretrained=True,
                                           num_classes=num_classes)
        except:
            inception_model = create_model('inception_resnet_v2',
                                           pretrained=False,
                                           num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Inception Architecture: {inception_type}')

    return inception_model
