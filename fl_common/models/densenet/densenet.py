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

def get_densenet_model(densenet_type, num_classes):
    """
    Obtain a DenseNet model with a specified architecture type and modify it for the given number of classes.

    Args:
    - densenet_type (str): Type of DenseNet architecture to be loaded.
      Options: 'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201'.
      Default is 'DenseNet121'.
    - num_classes (int): Number of output classes for the modified model. Default is 1000.

    Returns:
    - densenet_model (torch.nn.Module): The modified DenseNet model.

    Raises:
    - ValueError: If the provided densenet_type is not recognized.

    Note:
    - This function loads a pre-trained DenseNet model and modifies its last fully connected layer
      to match the specified number of output classes.

    Example Usage:
    ```python
    # Obtain a DenseNet121 model with 10 output classes
    model = get_densenet_model(densenet_type='DenseNet121', num_classes=10)
    ```
    """

    # Load the pre-trained version of DenseNet
    if densenet_type == 'DenseNet121':
        weights = models.DenseNet121_Weights.DEFAULT
        densenet_model = models.densenet121(weights=weights)
    elif densenet_type == 'DenseNet161':
        weights = models.DenseNet161_Weights.DEFAULT
        densenet_model = models.densenet161(weights=weights)
    elif densenet_type == 'DenseNet169':
        weights = models.DenseNet169_Weights.DEFAULT
        densenet_model = models.densenet169(weights=weights)
    elif densenet_type == 'DenseNet201':
        weights = models.DenseNet201_Weights.DEFAULT
        densenet_model = models.densenet201(weights=weights)
    else:
        raise ValueError(f'Unknown DenseNet Architecture : {densenet_type}')

    # Modify last layer to suit number of classes
    num_features = densenet_model.classifier.in_features
    densenet_model.classifier = nn.Linear(num_features, num_classes)

    return densenet_model
