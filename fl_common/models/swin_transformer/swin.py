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

def get_swin_model(swin_type, num_classes):
    """
    Loads a pre-trained Swin Transformer model based on the specified Swin type
    and modifies the last layer for a given number of output classes.

    Parameters:
    - swin_type (str, optional): Type of Swin Transformer model. Default is 'Swin_T'.
    - num_classes (int, optional): Number of output classes. Default is 1000.

    Returns:
    - swin_model (torch.nn.Module): Modified Swin Transformer model with the last layer
      adjusted for the specified number of output classes.

    Raises:
    - ValueError: If the specified Swin type is unknown or if the model does not have
      a known structure with a linear last layer.
    """

    # Load the pre-trained version of DenseNet
    if swin_type == 'Swin_T':
        weights = models.Swin_T_Weights.DEFAULT
        swin_model = models.swin_t(weights=weights)
    elif swin_type == 'Swin_S':
        weights = models.Swin_S_Weights.DEFAULT
        swin_model = models.swin_s(weights=weights)
    elif swin_type == 'Swin_B':
        weights = models.Swin_B_Weights.DEFAULT
        swin_model = models.swin_b(weights=weights)
    elif swin_type == 'Swin_V2_T':
        weights = models.Swin_V2_T_Weights.DEFAULT
        swin_model = models.swin_v2_t(weights=weights)
    elif swin_type == 'Swin_V2_S':
        weights = models.Swin_V2_S_Weights.DEFAULT
        swin_model = models.swin_v2_s(weights=weights)
    elif swin_type == 'Swin_V2_B':
        weights = models.Swin_V2_B_Weights.DEFAULT
        swin_model = models.swin_v2_b(weights=weights)
    else:
        raise ValueError(f'Unknown DenseNet Architecture : {swin_type}')

    # Modify last layer to suit number of classes
    if hasattr(swin_model, 'head') and isinstance(swin_model.head, nn.Linear):
        num_features = swin_model.head.in_features
        swin_model.head = nn.Linear(num_features, num_classes)
    else:
        raise ValueError('Model does not have a known structure.')

    return swin_model
