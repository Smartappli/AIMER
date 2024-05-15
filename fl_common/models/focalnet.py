from timm import create_model


def get_focalnet_model(focalnet_type, num_classes):
    """
    Creates and returns a Focalnet model based on the specified architecture type.

    Parameters:
        focalnet_type (str): Type of Focalnet architecture to use. Options include variations like
                             "focalnet_tiny_srf", "focalnet_small_srf", "focalnet_base_srf",
                             "focalnet_tiny_lrf", "focalnet_small_lrf", "focalnet_base_lrf",
                             "focalnet_large_fl3", "focalnet_large_fl4", "focalnet_xlarge_fl3",
                             "focalnet_xlarge_fl4", "focalnet_huge_fl3", "focalnet_huge_fl4".
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: A Focalnet model instance based on the specified architecture type.

    Raises:
        ValueError: If an unknown Focalnet architecture type is specified.
    """
    supported_types = {
        'focalnet_tiny_srf', 'focalnet_small_srf', 'focalnet_base_srf',
        'focalnet_tiny_lrf', 'focalnet_small_lrf', 'focalnet_base_lrf',
        'focalnet_large_fl3', 'focalnet_large_fl4', 'focalnet_xlarge_fl3',
        'focalnet_xlarge_fl4', 'focalnet_huge_fl3', 'focalnet_huge_fl4'
    }

    if focalnet_type not in supported_types:
        raise ValueError(f'Unknown Focalnet Architecture: {focalnet_type}')

    try:
        return create_model(
            focalnet_type,
            pretrained=True,
            num_classes=num_classes)
    except RuntimeError as e:
        print(f"{focalnet_type} - Error loading pretrained model: {e}")
        return create_model(
            focalnet_type,
            pretrained=False,
            num_classes=num_classes)
