from timm import create_model


def get_deit_model(deit_type, num_classes):
    """
    Get a DeiT (Data-efficient image Transformer) model.

    Parameters:
        deit_type (str): Type of DeiT architecture.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: DeiT model.

    Raises:
        ValueError: If an unknown Deit architecture is specified.
    """
    valid_deit_types = [
        "deit_tiny_patch16_224",
        "deit_small_patch16_224",
        "deit_base_patch16_224",
        "deit_base_patch16_384",
        "deit_tiny_distilled_patch16_224",
        "deit_small_distilled_patch16_224",
        "deit_base_distilled_patch16_224",
        "deit_base_distilled_patch16_384",
        "deit3_small_patch16_224",
        "deit3_small_patch16_384",
        "deit3_medium_patch16_224",
        "deit3_base_patch16_224",
        "deit3_base_patch16_384",
        "deit3_large_patch16_224",
        "deit3_large_patch16_384",
        "deit3_huge_patch14_224",
    ]

    if deit_type not in valid_deit_types:
        msg = f"Unknown Deit Architecture: {deit_type}"
        raise ValueError(msg)

    try:
        return create_model(deit_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{deit_type} - Error loading pretrained model: {e}")
        return create_model(
            deit_type, pretrained=False, num_classes=num_classes
        )
