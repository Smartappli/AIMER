from timm import create_model


def get_vision_transformer_hybrid_model(vision_transformer_hybrid_type, num_classes):
    """
    Retrieves a Vision Transformer Hybrid model based on the specified architecture.

    Args:
        vision_transformer_hybrid_type (str): Type of Vision Transformer Hybrid architecture.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Vision Transformer Hybrid model instance.

    Raises:
        ValueError: If an unsupported vision_transformer_hybrid_type is provided.
    """
    valid_types = {
        "vit_tiny_r_s16_p8_224",
        "vit_tiny_r_s16_p8_384",
        "vit_small_r26_s32_224",
        "vit_small_r26_s32_384",
        "vit_base_r26_s32_224",
        "vit_base_r50_s16_224",
        "vit_base_r50_s16_384",
        "vit_large_r50_s32_224",
        "vit_large_r50_s32_384",
        "vit_small_resnet26d_224",
        "vit_small_resnet50d_s16_224",
        "vit_base_resnet26d_224",
        "vit_base_resnet50d_224",
    }

    if vision_transformer_hybrid_type not in valid_types:
        raise ValueError(
            f"Unknown Vision Transformer Hybrid Architecture: {vision_transformer_hybrid_type}"
        )

    try:
        return create_model(
            vision_transformer_hybrid_type, pretrained=True, num_classes=num_classes
        )
    except RuntimeError as e:
        print(f"{vision_transformer_hybrid_type} - Error loading pretrained model: {e}")
        return create_model(
            vision_transformer_hybrid_type, pretrained=False, num_classes=num_classes
        )
