import torch.nn as nn
from torchvision import models


def get_vision_model(vision_type, num_classes):
    """
    Returns a modified Vision Transformer (ViT) model based on the specified type.

    Parameters:
        - vision_type (str): Type of Vision Transformer architecture.
                            Currently supports 'ViT_B_16', 'ViT_B_32', 'ViT_L_16', 'ViT_L_32', and 'ViT_H_14'.
        - num_classes (int): Number of classes for the modified last layer.

    Returns:
        - torch.nn.Module: Modified ViT model with the specified number of classes.

    Raises:
        - ValueError: If an unknown Vision Transformer architecture is provided.
    """
    # Load the pre-trained version of Vision Transformer based on the specified type
    if vision_type == 'ViT_B_16':
        weights = models.ViT_B_16_Weights.DEFAULT
        vision_model = models.vit_b_16(weights=weights)
    elif vision_type == 'ViT_B_32':
        weights = models.ViT_B_32_Weights.DEFAULT
        vision_model = models.vit_b_32(weights=weights)
    elif vision_type == 'ViT_L_16':
        weights = models.ViT_L_16_Weights.DEFAULT
        vision_model = models.vit_l_16(weights=weights)
    elif vision_type == 'ViT_L_32':
        weights = models.ViT_L_32_Weights.DEFAULT
        vision_model = models.vit_l_32(weights=weights)
    elif vision_type == 'ViT_H_14':
        weights = models.ViT_H_14_Weights.DEFAULT
        vision_model = models.vit_h_14(weights=weights)
    else:
        raise ValueError(f'Unknown Vision Transformer Architecture: {vision_type}')

    # Replace the last layer with a new linear layer with the specified number of classes
    last_layer = nn.Linear(in_features=vision_model.heads[-1].in_features, out_features=num_classes)
    vision_model.heads[-1] = last_layer

    return vision_model
