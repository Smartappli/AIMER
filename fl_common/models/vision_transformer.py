import torch.nn as nn
from torchvision import models
from timm import create_model


def get_vision_transformer_model(vision_type, num_classes):
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
    torch_vision = False
    # Load the pre-trained version of Vision Transformer based on the specified type
    if vision_type == 'ViT_B_16':
        try:
            weights = models.ViT_B_16_Weights.DEFAULT
            vision_model = models.vit_b_16(weights=weights)
        except:
            vision_model = models.vit_b_16(weights=None)

        torch_vision = True
    elif vision_type == 'ViT_B_32':
        try:
            weights = models.ViT_B_32_Weights.DEFAULT
            vision_model = models.vit_b_32(weights=weights)
        except:
            vision_model = models.vit_b_32(weights=None)

        torch_vision = True
    elif vision_type == 'ViT_L_16':
        try:
            weights = models.ViT_L_16_Weights.DEFAULT
            vision_model = models.vit_l_16(weights=weights)
        except:
            vision_model = models.vit_b_16(weights=None)

        torch_vision = True
    elif vision_type == 'ViT_L_32':
        try:
            weights = models.ViT_L_32_Weights.DEFAULT
            vision_model = models.vit_l_32(weights=weights)
        except:
            vision_model = models.vit_l_32(weights=None)

        torch_vision = True
    elif vision_type == 'ViT_H_14':
        try:
            weights = models.ViT_H_14_Weights.DEFAULT
            vision_model = models.vit_h_14(weights=weights)
        except:
            vision_model = models.vit_h_14(weights=None)

        torch_vision = True
    elif vision_type == "vit_tiny_patch16_224":
        try:
            vision_model = create_model('vit_tiny_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_tiny_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_tiny_patch16_384":
        try:
            vision_model = create_model('vit_tiny_patch16_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_tiny_patch16_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_small_patch32_224":
        try:
            vision_model = create_model('vit_small_patch32_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_small_patch32_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_small_patch32_384":
        try:
            vision_model = create_model('vit_small_patch32_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_small_patch32_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_small_patch16_224":
        try:
            vision_model = create_model('vit_small_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_small_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_small_patch16_384":
        try:
            vision_model = create_model('vit_small_patch16_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_small_patch16_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_small_patch8_224":
        try:
            vision_model = create_model('vit_small_patch8_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_small_patch8_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch32_224":
        try:
            vision_model = create_model('vit_base_patch32_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch32_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch32_384":
        try:
            vision_model = create_model('vit_base_patch32_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch32_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch16_224":
        try:
            vision_model = create_model('vit_base_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch16_384":
        try:
            vision_model = create_model('vit_base_patch16_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch16_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch8_224":
        try:
            vision_model = create_model('vit_base_patch8_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch8_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_large_patch32_224":
        try:
            vision_model = create_model('vit_large_patch32_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_large_patch32_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_large_patch32_384":
        try:
            vision_model = create_model('vit_large_patch32_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_large_patch32_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_large_patch16_224":
        try:
            vision_model = create_model('vit_large_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_large_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_large_patch16_384":
        try:
            vision_model = create_model('vit_large_patch16_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_large_patch16_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_large_patch14_224":
        try:
            vision_model = create_model('vit_large_patch14_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_large_patch14_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_large_patch14_224":
        try:
            vision_model = create_model('vit_huge_patch14_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_huge_patch14_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_giant_patch14_224":
        try:
            vision_model = create_model('vit_giant_patch14_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_giant_patch14_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_gigantic_patch14_224":
        try:
            vision_model = create_model('vit_gigantic_patch14_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_gigantic_patch14_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch16_224_miil":
        try:
            vision_model = create_model('vit_base_patch16_224_miil',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch16_224_miil',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_medium_patch16_gap_240":
        try:
            vision_model = create_model('vit_medium_patch16_gap_240',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_medium_patch16_gap_240',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_medium_patch16_gap_256":
        try:
            vision_model = create_model('vit_medium_patch16_gap_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_medium_patch16_gap_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_medium_patch16_gap_384":
        try:
            vision_model = create_model('vit_medium_patch16_gap_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_medium_patch16_gap_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch16_gap_224":
        try:
            vision_model = create_model('vit_base_patch16_gap_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch16_gap_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_huge_patch14_gap_224":
        try:
            vision_model = create_model('vit_huge_patch14_gap_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_huge_patch14_gap_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_huge_patch16_gap_448":
        try:
            vision_model = create_model('vit_huge_patch16_gap_448',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_huge_patch16_gap_448',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_giant_patch16_gap_224":
        try:
            vision_model = create_model('vit_giant_patch16_gap_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_giant_patch16_gap_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_xsmall_patch16_clip_224":
        try:
            vision_model = create_model('vit_xsmall_patch16_clip_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_xsmall_patch16_clip_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_medium_patch32_clip_224":
        try:
            vision_model = create_model('vit_medium_patch32_clip_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_medium_patch32_clip_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_medium_patch16_clip_224":
        try:
            vision_model = create_model('vit_medium_patch16_clip_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_medium_patch16_clip_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_betwixt_patch32_clip_224":
        try:
            vision_model = create_model('vit_betwixt_patch32_clip_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_betwixt_patch32_clip_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch32_clip_224":
        try:
            vision_model = create_model('vit_base_patch32_clip_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch32_clip_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch32_clip_256":
        try:
            vision_model = create_model('vit_base_patch32_clip_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch32_clip_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch32_clip_384":
        try:
            vision_model = create_model('vit_base_patch32_clip_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch32_clip_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch32_clip_448":
        try:
            vision_model = create_model('vit_base_patch32_clip_448',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch32_clip_448',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch16_clip_224":
        try:
            vision_model = create_model('vit_base_patch16_clip_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch16_clip_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch16_clip_384":
        try:
            vision_model = create_model('vit_base_patch16_clip_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch16_clip_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_large_patch14_clip_224":
        try:
            vision_model = create_model('vit_large_patch14_clip_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_large_patch14_clip_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_large_patch14_clip_336":
        try:
            vision_model = create_model('vit_large_patch14_clip_336',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_large_patch14_clip_336',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_huge_patch14_clip_224":
        try:
            vision_model = create_model('vit_huge_patch14_clip_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_huge_patch14_clip_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_huge_patch14_clip_336":
        try:
            vision_model = create_model('vit_huge_patch14_clip_336',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_huge_patch14_clip_336',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_huge_patch14_clip_378":
        try:
            vision_model = create_model('vit_huge_patch14_clip_378',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_huge_patch14_clip_378',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_giant_patch14_clip_224":
        try:
            vision_model = create_model('vit_giant_patch14_clip_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_giant_patch14_clip_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_gigantic_patch14_clip_224":
        try:
            vision_model = create_model('vit_gigantic_patch14_clip_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_gigantic_patch14_clip_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch32_clip_quickgelu_224":
        try:
            vision_model = create_model('vit_base_patch32_clip_quickgelu_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch32_clip_quickgelu_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch16_clip_quickgelu_224":
        try:
            vision_model = create_model('vit_base_patch16_clip_quickgelu_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch16_clip_quickgelu_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_large_patch14_clip_quickgelu_224":
        try:
            vision_model = create_model('vit_large_patch14_clip_quickgelu_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_large_patch14_clip_quickgelu_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_large_patch14_clip_quickgelu_336":
        try:
            vision_model = create_model('vit_large_patch14_clip_quickgelu_336',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_large_patch14_clip_quickgelu_336',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_huge_patch14_clip_quickgelu_224":
        try:
            vision_model = create_model('vit_huge_patch14_clip_quickgelu_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_huge_patch14_clip_quickgelu_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_huge_patch14_clip_quickgelu_378":
        try:
            vision_model = create_model('vit_huge_patch14_clip_quickgelu_378',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_huge_patch14_clip_quickgelu_378',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch32_plus_256":
        try:
            vision_model = create_model('vit_base_patch32_plus_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch32_plus_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch16_plus_240":
        try:
            vision_model = create_model('vit_base_patch16_plus_240',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch16_plus_240',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch16_rpn_224":
        try:
            vision_model = create_model('vit_base_patch16_rpn_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch16_rpn_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_small_patch16_36x1_224":
        try:
            vision_model = create_model('vit_small_patch16_36x1_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_small_patch16_36x1_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_small_patch16_18x2_224":
        try:
            vision_model = create_model('vit_small_patch16_18x2_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_small_patch16_18x2_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch16_18x2_224":
        try:
            vision_model = create_model('vit_base_patch16_18x2_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch16_18x2_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "eva_large_patch14_196":
        try:
            vision_model = create_model('eva_large_patch14_196',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('eva_large_patch14_196',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "eva_large_patch14_336":
        try:
            vision_model = create_model('eva_large_patch14_336',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('eva_large_patch14_336',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "flexivit_small":
        try:
            vision_model = create_model('flexivit_small',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('flexivit_small',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "flexivit_base":
        try:
            vision_model = create_model('flexivit_base',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('flexivit_base',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "flexivit_large":
        try:
            vision_model = create_model('flexivit_large',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('flexivit_large',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch16_xp_224":
        try:
            vision_model = create_model('vit_base_patch16_xp_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch16_xp_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_large_patch14_xp_224":
        try:
            vision_model = create_model('vit_large_patch14_xp_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_large_patch14_xp_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_huge_patch14_xp_224":
        try:
            vision_model = create_model('vit_huge_patch14_xp_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_huge_patch14_xp_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_small_patch14_dinov2":
        try:
            vision_model = create_model('vit_small_patch14_dinov2',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_small_patch14_dinov2',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch14_dinov2":
        try:
            vision_model = create_model('vit_base_patch14_dinov2',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch14_dinov2',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_large_patch14_dinov2":
        try:
            vision_model = create_model('vit_large_patch14_dinov2',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_large_patch14_dinov2',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_giant_patch14_dinov2":
        try:
            vision_model = create_model('vit_giant_patch14_dinov2',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_giant_patch14_dinov2',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_small_patch14_reg4_dinov2":
        try:
            vision_model = create_model('vit_small_patch14_reg4_dinov2',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_small_patch14_reg4_dinov2',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch14_reg4_dinov2":
        try:
            vision_model = create_model('vit_base_patch14_reg4_dinov2',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch14_reg4_dinov2',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_large_patch14_reg4_dinov2":
        try:
            vision_model = create_model('vit_large_patch14_reg4_dinov2',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_large_patch14_reg4_dinov2',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_giant_patch14_reg4_dinov2":
        try:
            vision_model = create_model('vit_giant_patch14_reg4_dinov2',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_giant_patch14_reg4_dinov2',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch16_siglip_224":
        try:
            vision_model = create_model('vit_base_patch16_siglip_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch16_siglip_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch16_siglip_256":
        try:
            vision_model = create_model('vit_base_patch16_siglip_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch16_siglip_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch16_siglip_384":
        try:
            vision_model = create_model('vit_base_patch16_siglip_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch16_siglip_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch16_siglip_512":
        try:
            vision_model = create_model('vit_base_patch16_siglip_512',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch16_siglip_512',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_large_patch16_siglip_256":
        try:
            vision_model = create_model('vit_large_patch16_siglip_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_large_patch16_siglip_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_large_patch16_siglip_384":
        try:
            vision_model = create_model('vit_large_patch16_siglip_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_large_patch16_siglip_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_so400m_patch14_siglip_224":
        try:
            vision_model = create_model('vit_so400m_patch14_siglip_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_so400m_patch14_siglip_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_so400m_patch14_siglip_384":
        try:
            vision_model = create_model('vit_so400m_patch14_siglip_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_so400m_patch14_siglip_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_medium_patch16_reg4_256":
        try:
            vision_model = create_model('vit_medium_patch16_reg4_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_medium_patch16_reg4_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_medium_patch16_reg4_gap_256":
        try:
            vision_model = create_model('vit_medium_patch16_reg4_gap_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_medium_patch16_reg4_gap_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_base_patch16_reg4_gap_256":
        try:
            vision_model = create_model('vit_base_patch16_reg4_gap_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_base_patch16_reg4_gap_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_so150m_patch16_reg4_map_256":
        try:
            vision_model = create_model('vit_so150m_patch16_reg4_map_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_so150m_patch16_reg4_map_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif vision_type == "vit_so150m_patch16_reg4_gap_256":
        try:
            vision_model = create_model('vit_so150m_patch16_reg4_gap_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            vision_model = create_model('vit_so150m_patch16_reg4_gap_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Vision Transformer Architecture: {vision_type}')

    if torch_vision:
        # Replace the last layer with a new linear layer with the specified number of classes
        last_layer = nn.Linear(in_features=vision_model.heads[-1].in_features, out_features=num_classes)
        vision_model.heads[-1] = last_layer

    return vision_model
