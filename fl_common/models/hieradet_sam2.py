from timm import create_model


def get_hiera_sam2_model(hiera_sam2_type, num_classes):
    """
    Create and return a HIERA SAM2 model based on the specified architecture.

    Args:
        hiera_sam2_type (str): Type of HIERA SAM2 architecture.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Created HIERA SAM2 model.

    Raises:
        ValueError: If an unknown HIERA SAM2 architecture is specified.
    """
    valid_hiera_sam2_types = {
        "sam2_hiera_tiny",
        "sam2_hiera_small",
        "sam2_hiera_base_plus",
        "sam2_hiera_large",
        "hieradet_small",
    }

    if hiera_sam2_type not in valid_hiera_sam2_types:
        msg = f"Unknown HIERA SAM2 Architecture: {hiera_sam2_type}"
        raise ValueError(msg)

    try:
        return create_model(
            hiera_sam2_type,
            pretrained=True,
            num_classes=num_classes,
        )
    except RuntimeError as e:
        print(f"{hiera_sam2_type} - Error loading pretrained model: {e}")
        return create_model(
            hiera_sam2_type,
            pretrained=False,
            num_classes=num_classes,
        )
