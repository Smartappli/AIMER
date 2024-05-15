import torch.nn as nn
from torchvision import models


def get_shufflenet_model(shufflenet_type, num_classes):
    """
    Load a pre-trained ShuffleNet model of the specified type and modify its
    last layer to accommodate the given number of classes.
    """

    # Dictionary mapping model types to their respective functions and weights
    model_funcs = {
        'ShuffleNet_V2_X0_5': (models.shufflenet_v2_x0_5, models.ShuffleNet_V2_X0_5_Weights),
        'ShuffleNet_V2_X1_0': (models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights),
        'ShuffleNet_V2_X1_5': (models.shufflenet_v2_x1_5, models.ShuffleNet_V2_X1_5_Weights),
        'ShuffleNet_V2_X2_0': (models.shufflenet_v2_x2_0, models.ShuffleNet_V2_X2_0_Weights)
    }

    # Validate shufflenet_type
    if shufflenet_type not in model_funcs:
        raise ValueError(f'Unknown ShuffleNet Architecture: {shufflenet_type}')

    # Load the pre-trained model
    model_func, weights = model_funcs[shufflenet_type]
    try:
        shufflenet_model = model_func(weights=weights.DEFAULT)
    except RuntimeError as e:
        print(f"{shufflenet_type} - Error loading pretrained model: {e}")
        shufflenet_model = model_func(weights=None)

    # Modify last layer to suit number of classes
    num_features = shufflenet_model.fc.in_features
    shufflenet_model.fc = nn.Linear(num_features, num_classes)

    return shufflenet_model
