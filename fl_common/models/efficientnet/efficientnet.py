import os
import time
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from captum.attr import (
    Saliency,
    IntegratedGradients,
    GuidedBackprop,
    DeepLift,
    LayerConductance,
    NeuronConductance,
    Occlusion,
    ShapleyValueSampling,
)
from sklearn.metrics import confusion_matrix, classification_report
# from torch.utils.data.sampler import SubsetRandomSampler
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from fl_common.models.utils import (get_optimizer,
                                    get_criterion,
                                    get_scheduler,
                                    generate_xai_heatmaps,
                                    get_dataset,
                                    EarlyStopping)

def get_efficientnet_model(efficientnet_type='EfficientNetB0', num_classes=1000):
    """
    Obtain an EfficientNet model with a specified architecture type and modify it for the given number of classes.

    Args:
    - efficientnet_type (str): Type of EfficientNet architecture to be loaded.
      Options: 'EfficientNetB0' to 'EfficientNetB7', 'EfficientNetV2S', 'EfficientNetV2M', 'EfficientNetV2L'.
      Default is 'EfficientNetB0'.
    - num_classes (int): Number of output classes for the modified model. Default is 1000.

    Returns:
    - efficientnet_model (torch.nn.Module): The modified EfficientNet model.

    Raises:
    - ValueError: If the provided efficientnet_type is not recognized.

    Note:
    - This function loads a pre-trained EfficientNet model and modifies its last fully connected layer
      to match the specified number of output classes.

    Example Usage:
    ```python
    # Obtain an EfficientNetB0 model with 10 output classes
    model = get_efficientnet_model(efficientnet_type='EfficientNetB0', num_classes=10)
    ```
    """

    # Load the pre-trained version of EfficientNet
    if efficientnet_type == 'EfficientNetB0':
        weights = models.EfficientNet_B0_Weights.DEFAULT
        efficientnet_model = models.efficientnet_b0(weights=weights)
    elif efficientnet_type == 'EfficientNetB1':
        weights = models.EfficientNet_B1_Weights.DEFAULT
        efficientnet_model = models.efficientnet_b1(weights=weights)
    elif efficientnet_type == 'EfficientNetB2':
        weights = models.EfficientNet_B2_Weights.DEFAULT
        efficientnet_model = models.efficientnet_b2(weights=weights)
    elif efficientnet_type == 'EfficientNetB3':
        weights = models.EfficientNet_B3_Weights.DEFAULT
        efficientnet_model = models.efficientnet_b3(weights=weights)
    elif efficientnet_type == 'EfficientNetB4':
        weights = models.EfficientNet_B4_Weights.DEFAULT
        efficientnet_model = models.efficientnet_b4(weights=weights)
    elif efficientnet_type == 'EfficientNetB5':
        weights = models.EfficientNet_B5_Weights.DEFAULT
        efficientnet_model = models.efficientnet_b5(weights=weights)
    elif efficientnet_type == 'EfficientNetB6':
        weights = models.EfficientNet_B6_Weights.DEFAULT
        efficientnet_model = models.efficientnet_b6(weights=weights)
    elif efficientnet_type == 'EfficientNetB7':
        weights = models.EfficientNet_B7_Weights.DEFAULT
        efficientnet_model = models.efficientnet_b7(weights=weights)
    elif efficientnet_type == 'EfficientNetV2S':
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        efficientnet_model = models.efficientnet_v2_s(weights=weights)
    elif efficientnet_type == 'EfficientNetV2M':
        weights = models.EfficientNet_V2_M_Weights.DEFAULT
        efficientnet_model = models.efficientnet_v2_m(weights=weights)
    elif efficientnet_type == 'EfficientNetV2L':
        weights = models.EfficientNet_V2_L_Weights.DEFAULT
        efficientnet_model = models.efficientnet_v2_l(weights=weights)
    else:
        raise ValueError(f'Unknown DenseNet Architecture : {efficientnet_type}')

    # Modify last layer to suit number of classes
    num_features = efficientnet_model.classifier[-1].in_features
    efficientnet_model.classifier[-1] = nn.Linear(num_features, num_classes)

    return efficientnet_model
