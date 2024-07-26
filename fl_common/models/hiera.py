from timm import create_model


def get_hiera_model(hiera_type, num_classes):
    """
    Create and return a HIERA model based on the specified architecture.

    Args:
        hiera_type (str): Type of HIERA architecture.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Created HIERA model.

    Raises:
        ValueError: If an unknown HIERA architecture is specified.
    """
    valid_hiera_types = {
        "hiera_tiny_224",
        "hiera_small_224",
        "hiera_base_224",
        "hiera_base_plus_224",
        "hiera_large_224",
        "hiera_huge_224",
        "hiera_small_abswin_256",
        "hiera_base_abswin_256",
    }

    if hiera_type not in valid_hiera_types:
        msg = f"Unknown HIERA Architecture: {hiera_type}"
        raise ValueError(msg)

    try:
        return create_model(
            hiera_type,
            pretrained=True,
            num_classes=num_classes,
        )
    except RuntimeError as e:
        print(f"{hiera_type} - Error loading pretrained model: {e}")
        return create_model(
            hiera_type,
            pretrained=False,
            num_classes=num_classes,
        )
