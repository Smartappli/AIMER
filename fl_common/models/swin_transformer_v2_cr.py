from timm import create_model


def get_swin_transformer_v2_cr_model(swin_type, num_classes):
    """
    Get a Swin Transformer v2 model for classification/regression.

    Args:
        swin_type (str): Type of Swin Transformer v2 model. Options include:
            - 'swinv2_cr_tiny_384'
            - 'swinv2_cr_tiny_224'
            - 'swinv2_cr_tiny_ns_224'
            - 'swinv2_cr_small_384'
            - 'swinv2_cr_small_224'
            - 'swinv2_cr_small_ns_224'
            - 'swinv2_cr_small_ns_256'
            - 'swinv2_cr_base_384'
            - 'swinv2_cr_base_224'
            - 'swinv2_cr_base_ns_224'
            - 'swinv2_cr_large_384'
            - 'swinv2_cr_large_224'
            - 'swinv2_cr_huge_384'
            - 'swinv2_cr_huge_224'
            - 'swinv2_cr_giant_384'
            - 'swinv2_cr_giant_224'
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Swin Transformer v2 model.
    """
    supported_types = {
        "swinv2_cr_tiny_384",
        "swinv2_cr_tiny_224",
        "swinv2_cr_tiny_ns_224",
        "swinv2_cr_small_384",
        "swinv2_cr_small_224",
        "swinv2_cr_small_ns_224",
        "swinv2_cr_small_ns_256",
        "swinv2_cr_base_384",
        "swinv2_cr_base_224",
        "swinv2_cr_base_ns_224",
        "swinv2_cr_large_384",
        "swinv2_cr_large_224",
        "swinv2_cr_huge_384",
        "swinv2_cr_huge_224",
        "swinv2_cr_giant_384",
        "swinv2_cr_giant_224",
    }

    if swin_type not in supported_types:
        msg = f"Unknown Swin Transformer v2 cr Architecture: {swin_type}"
        raise ValueError(msg)

    try:
        return create_model(swin_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{swin_type} - Error loading pretrained model: {e}")
        return create_model(
            swin_type,
            pretrained=False,
            num_classes=num_classes,
        )
