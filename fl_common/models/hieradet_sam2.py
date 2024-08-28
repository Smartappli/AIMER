from timm import create_model


def get_hieradet_sam2_model(hieradet_sam2_type, num_classes):
    """
    Create and return a HIERADET SAM2 model based on the specified architecture.

    Args:
        hiera_sam2_type (str): Type of HIERADET SAM2 architecture.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Created HIERADET SAM2 model.

    Raises:
        ValueError: If an unknown HIERADET SAM2 architecture is specified.
    """
    valid_hieradet_sam2_types = {
        "sam2_hiera_tiny",
        "sam2_hiera_small",
        "sam2_hiera_base_plus",
        "sam2_hiera_large",
        "hieradet_small",
    }

    if hieradet_sam2_type not in valid_hieradet_sam2_types:
        msg = f"Unknown HIERADET SAM2 Architecture: {hieradet_sam2_type}"
        raise ValueError(msg)

    try:
        return create_model(
            hieradet_sam2_type,
            pretrained=True,
            num_classes=num_classes,
        )
    except RuntimeError as e:
        print(f"{hieradet_sam2_type} - Error loading pretrained model: {e}")
        return create_model(
            hieradet_sam2_type,
            pretrained=False,
            num_classes=num_classes,
        )
