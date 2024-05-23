from torch import nn
from torchvision import models
from timm import create_model


def get_vision_transformer_model(vision_type, num_classes):
    """
    Returns a modified Vision Transformer (ViT) model based on the specified type.

    Parameters:
        - vision_type (str): Type of Vision Transformer architecture.
            Currently, supports 'ViT_B_16', 'ViT_B_32', 'ViT_L_16', 'ViT_L_32', 'ViT_H_14',
            and various models from the 'timm' library.
        - num_classes (int): Number of classes for the modified last layer.

    Returns:
        - torch.nn.Module: Modified ViT model with the specified number of classes.

    Raises:
        - ValueError: If an unknown Vision Transformer architecture is provided.
    """
    # Mapping of vision types to their corresponding torchvision models and
    # weights
    torchvision_models = {
        "ViT_B_16": (models.vit_b_16, models.ViT_B_16_Weights),
        "ViT_B_32": (models.vit_b_32, models.ViT_B_32_Weights),
        "ViT_L_16": (models.vit_l_16, models.ViT_L_16_Weights),
        "ViT_L_32": (models.vit_l_32, models.ViT_L_32_Weights),
        "ViT_H_14": (models.vit_h_14, models.ViT_H_14_Weights),
    }

    timm_models = [
        "vit_tiny_patch16_224",
        "vit_tiny_patch16_384",
        "vit_small_patch32_224",
        "vit_small_patch32_384",
        "vit_small_patch16_224",
        "vit_small_patch16_384",
        "vit_small_patch8_224",
        "vit_base_patch32_224",
        "vit_base_patch32_384",
        "vit_base_patch16_224",
        "vit_base_patch16_384",
        "vit_base_patch8_224",
        "vit_large_patch32_224",
        "vit_large_patch32_384",
        "vit_large_patch16_224",
        "vit_large_patch16_384",
        "vit_large_patch14_224",
        "vit_large_patch14_224",
        "vit_giant_patch14_224",
        "vit_gigantic_patch14_224",
        "vit_base_patch16_224_miil",
        "vit_medium_patch16_gap_240",
        "vit_medium_patch16_gap_256",
        "vit_medium_patch16_gap_384",
        "vit_base_patch16_gap_224",
        "vit_huge_patch14_gap_224",
        "vit_huge_patch16_gap_448",
        "vit_giant_patch16_gap_224",
        "vit_xsmall_patch16_clip_224",
        "vit_medium_patch32_clip_224",
        "vit_medium_patch16_clip_224",
        "vit_betwixt_patch32_clip_224",
        "vit_base_patch32_clip_224",
        "vit_base_patch32_clip_256",
        "vit_base_patch32_clip_384",
        "vit_base_patch32_clip_448",
        "vit_base_patch16_clip_224",
        "vit_base_patch16_clip_384",
        "vit_large_patch14_clip_224",
        "vit_large_patch14_clip_336",
        "vit_huge_patch14_clip_224",
        "vit_huge_patch14_clip_336",
        "vit_huge_patch14_clip_378",
        "vit_giant_patch14_clip_224",
        "vit_gigantic_patch14_clip_224",
        "vit_base_patch32_clip_quickgelu_224",
        "vit_base_patch16_clip_quickgelu_224",
        "vit_large_patch14_clip_quickgelu_224",
        "vit_large_patch14_clip_quickgelu_336",
        "vit_huge_patch14_clip_quickgelu_224",
        "vit_huge_patch14_clip_quickgelu_378",
        "vit_base_patch32_plus_256",
        "vit_base_patch16_plus_240",
        "vit_base_patch16_rpn_224",
        "vit_small_patch16_36x1_224",
        "vit_small_patch16_18x2_224",
        "vit_base_patch16_18x2_224",
        "eva_large_patch14_196",
        "eva_large_patch14_336",
        "flexivit_small",
        "flexivit_base",
        "flexivit_large",
        "vit_base_patch16_xp_224",
        "vit_large_patch14_xp_224",
        "vit_huge_patch14_xp_224",
        "vit_small_patch14_dinov2",
        "vit_base_patch14_dinov2",
        "vit_large_patch14_dinov2",
        "vit_giant_patch14_dinov2",
        "vit_small_patch14_reg4_dinov2",
        "vit_base_patch14_reg4_dinov2",
        "vit_large_patch14_reg4_dinov2",
        "vit_giant_patch14_reg4_dinov2",
        "vit_base_patch16_siglip_224",
        "vit_base_patch16_siglip_256",
        "vit_base_patch16_siglip_384",
        "vit_base_patch16_siglip_512",
        "vit_large_patch16_siglip_256",
        "vit_large_patch16_siglip_384",
        "vit_so400m_patch14_siglip_224",
        "vit_so400m_patch14_siglip_384",
        "vit_base_patch16_siglip_gap_224",
        "vit_base_patch16_siglip_gap_256",
        "vit_base_patch16_siglip_gap_384",
        "vit_base_patch16_siglip_gap_512",
        "vit_large_patch16_siglip_gap_256",
        "vit_large_patch16_siglip_gap_384",
        "vit_so400m_patch14_siglip_gap_224",
        "vit_so400m_patch14_siglip_gap_384",
        "vit_so400m_patch14_siglip_gap_448",
        "vit_so400m_patch14_siglip_gap_896",
        "vit_wee_patch16_reg1_gap_256",
        "vit_pwee_patch16_reg1_gap_256",
        "vit_little_patch16_reg4_gap_256",
        "vit_medium_patch16_reg1_gap_256",
        "vit_medium_patch16_reg4_gap_256",
        "vit_mediumd_patch16_reg4_gap_256",
        "vit_betwixt_patch16_reg1_gap_256",
        "vit_betwixt_patch16_reg4_gap_256",
        "vit_base_patch16_reg4_gap_256",
        "vit_so150m_patch16_reg4_map_256",
        "vit_so150m_patch16_reg4_gap_256",
    ]

    # Check if the vision type is from torchvision
    if vision_type in torchvision_models:
        model_func, weights_class = torchvision_models[vision_type]
        try:
            weights = weights_class.DEFAULT
            vision_model = model_func(weights=weights)
        except RuntimeError as e:
            print(f"{vision_type} - Error loading pretrained model: {e}")
            vision_model = model_func(weights=None)

        # Replace the last layer with a new linear layer with the specified
        # number of classes
        last_layer = nn.Linear(
            in_features=vision_model.heads[-1].in_features, out_features=num_classes
        )
        vision_model.heads[-1] = last_layer

    # Check if the vision type is from the 'timm' library
    elif vision_type in timm_models:
        try:
            vision_model = create_model(
                vision_type, pretrained=True, num_classes=num_classes
            )
        except RuntimeError as e:
            print(f"{vision_type} - Error loading pretrained model: {e}")
            vision_model = create_model(
                vision_type, pretrained=False, num_classes=num_classes
            )
    else:
        raise ValueError(f"Unknown Vision Transformer Architecture: {vision_type}")

    return vision_model
