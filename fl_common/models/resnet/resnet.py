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

def get_resnet_model(resnet_type, num_classes):
    """
    Obtain a ResNet model with a specified architecture type and modify it for the given number of classes.

    Args:
    - resnet_type (str): Type of ResNet architecture to be loaded.
      Options: 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
      'ResNeXt50_32X4D', 'ResNeXt101_32X8D', 'ResNeXt101_64X4D', 'Wide_ResNet50_2', 'Wide_ResNet101_2'.
      Default is 'ResNet50'.
    - num_classes (int): Number of output classes for the modified model. Default is 1000.

    Returns:
    - resnet_model (torch.nn.Module): The modified ResNet model.

    Raises:
    - ValueError: If the provided resnet_type is not recognized.

    Note:
    - This function loads a pre-trained ResNet model and modifies its last fully connected layer
      to match the specified number of output classes.

    Example Usage:
    ```python
    # Obtain a ResNet50 model with 10 output classes
    model = get_resnet_model(resnet_type='ResNet50', num_classes=10)
    ```
    """

    # Load the pre-trained version of DenseNet
    if resnet_type == 'ResNet18':
        weights = models.ResNet18_Weights.DEFAULT
        resnet_model = models.resnet18(weights=weights)
    elif resnet_type == 'ResNet34':
        weights = models.ResNet34_Weights.DEFAULT
        resnet_model = models.resnet34(weights=weights)
    elif resnet_type == 'ResNet50':
        weights = models.ResNet50_Weights.DEFAULT
        resnet_model = models.resnet50(weights=weights)
    elif resnet_type == 'ResNet101':
        weights = models.ResNet101_Weights.DEFAULT
        resnet_model = models.resnet101(weights=weights)
    elif resnet_type == 'ResNet152':
        weights = models.ResNet152_Weights.DEFAULT
        resnet_model = models.resnet152(weights=weights)
    elif resnet_type == 'ResNeXt50_32X4D':
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT
        resnet_model = models.resnext50_32x4d(weights=weights)
    elif resnet_type == 'ResNeXt101_32X8D':
        weights = models.ResNeXt101_32X8D_Weights.DEFAULT
        resnet_model = models.resnext101_32x8d(weights=weights)
    elif resnet_type == 'ResNeXt101_64X4D':
        weights = models.ResNeXt101_64X4D_Weights.DEFAULT
        resnet_model = models.resnext101_64x4d(weights=weights)
    elif resnet_type == 'Wide_ResNet50_2':
        weights = models.Wide_ResNet50_2_Weights.DEFAULT
        resnet_model = models.wide_resnet50_2(weights=weights)
    elif resnet_type == 'Wide_ResNet101_2':
        weights = models.Wide_ResNet101_2_Weights.DEFAULT
        resnet_model = models.wide_resnet101_2(weights=weights)
    else:
        raise ValueError(f'Unknown DenseNet Architecture : {resnet_type}')

    # Modify last layer to suit number of classes
    num_features = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_features, num_classes)

    return resnet_model
