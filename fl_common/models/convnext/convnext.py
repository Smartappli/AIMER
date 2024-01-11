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

def get_convnext_model(convnext_type, num_classes):
    """0
    Obtain a ConvNeXt model with a specified architecture type and modify it for the given number of classes.

    Args:
    - convnext_type (str): Type of ConvNeXt architecture to be loaded. Options: 'ConvNeXt_Tiny', 'ConvNeXt_Small',
      'ConvNeXt_Base', 'ConvNeXt_Large'. Default is 'ConvNeXt_Large'.
    - num_classes (int): Number of output classes for the modified model. Default is 1000.

    Returns:
    - convnext_model (torch.nn.Module): The modified ConvNeXt model.

    Raises:
    - ValueError: If the provided convnext_type is not recognized or the model structure is not as expected.

    Note:
    - This function loads a pre-trained ConvNeXt model and modifies its last fully connected layer to
      match the specified number of output classes.

    Example Usage:
    ```python
    # Obtain a ConvNeXt model with 10 output classes
    model = get_convnext_model(convnext_type='ConvNeXt_Small', num_classes=10)
    ```
    """

    # Load the pre-trained version of DenseNet
    if convnext_type == 'ConvNeXt_Tiny':
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        convnext_model = models.convnext_tiny(weights=weights)
    elif convnext_type == 'ConvNeXt_Small':
        weights = models.ConvNeXt_Small_Weights.DEFAULT
        convnext_model = models.convnext_small(weights=weights)
    elif convnext_type == 'ConvNeXt_Base':
        weights = models.ConvNeXt_Base_Weights.DEFAULT
        convnext_model = models.convnext_base(weights=weights)
    elif convnext_type == 'ConvNeXt_Large':
        weights = models.ConvNeXt_Large_Weights.DEFAULT
        convnext_model = models.convnext_large(weights=weights)
    else:
        raise ValueError(f'Unknown DenseNet Architecture : {convnext_type}')

    # Modify last layer to suit number of classes
    for layer in reversed(convnext_model.classifier):
        if isinstance(layer, nn.Linear):
            # num_features = layer.in_features
            layer.out_features = num_classes
            break

    return convnext_model
