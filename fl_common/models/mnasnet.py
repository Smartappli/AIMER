import torch.nn as nn
from torchvision import models


def get_mnasnet_model(mnasnet_type, num_classes):
    """
    Load a pre-trained MNASNet model of the specified type and modify its
    last layer to accommodate the given number of classes.

    Parameters:
    - mnasnet_type (str): Type of MNASNet architecture, supported types:
        - 'MNASNet0_5'
        - 'MNASNet0_75'
        - 'MNASNet1_0'
        - 'MNASNet1_3'
    - num_classes (int): Number of output classes for the modified last layer.

    Returns:
    - torch.nn.Module: Modified MNASNet model with the specified architecture
      and last layer adapted for the given number of classes.

    Raises:
    - ValueError: If an unknown MNASNet architecture type is provided.
    """
    # Load the pre-trained version of MNASNet based on the specified type
    if mnasnet_type == 'MNASNet0_5':
        try:
            weights = models.MNASNet0_5_Weights.DEFAULT
            mnasnet_model = models.mnasnet.mnasnet0_5(weights=weights)
        except:
            mnasnet_model = models.mnasnet.mnasnet0_5(weights=None)
    elif mnasnet_type == 'MNASNet0_75':
        try:
            weights = models.MNASNet0_75_Weights.DEFAULT
            mnasnet_model = models.mnasnet.mnasnet0_75(weights=weights)
        except:
            mnasnet_model = models.mnasnet.mnasnet0_75(weights=None)
    elif mnasnet_type == 'MNASNet1_0':
        try:
            weights = models.MNASNet1_0_Weights.DEFAULT
            mnasnet_model = models.mnasnet.mnasnet1_0(weights=weights)
        except:
            mnanet_model = models.mnasnet.mnasnet1_0(weights=None)
    elif mnasnet_type == 'MNASNet1_3':
        try:
            weights = models.MNASNet1_3_Weights.DEFAULT
            mnasnet_model = models.mnasnet.mnasnet1_3(weights=weights)
        except:
            mnasnet_model = models.mnasnet.mnasnet1_3(weights=None)
    else:
        raise ValueError(f'Unknown MNASNet Architecture: {mnasnet_type}')

    # Modify the last layer to suit the given number of classes
    num_features = mnasnet_model.classifier[1].in_features
    mnasnet_model.classifier[1] = nn.Linear(num_features, num_classes)

    return mnasnet_model
