from timm import create_model


def get_hardcorenas_model(hardcorenas_type, num_classes):
    """
    Get a HardcoreNAS (Hardcore Neural Architecture Search) model.

    Parameters:
        hardcorenas_type (str): Type of HardcoreNAS architecture. Options include:
            - 'hardcorenas_a'
            - 'hardcorenas_b'
            - 'hardcorenas_c'
            - 'hardcorenas_d'
            - 'hardcorenas_e'
            - 'hardcorenas_f'
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: HardcoreNAS model.

    Raises:
        ValueError: If an unknown HardcoreNAS architecture is specified.
    """
    supported_types = {
        "hardcorenas_a",
        "hardcorenas_b",
        "hardcorenas_c",
        "hardcorenas_d",
        "hardcorenas_e",
        "hardcorenas_f",
    }

    if hardcorenas_type not in supported_types:
        msg = f"Unknown HardcoreNAS Architecture: {hardcorenas_type}"
        raise ValueError(msg)

    try:
        return create_model(
            hardcorenas_type,
            pretrained=True,
            num_classes=num_classes,
        )
    except RuntimeError as e:
        print(f"{hardcorenas_type} - Error loading pretrained model: {e}")
        return create_model(
            hardcorenas_type,
            pretrained=False,
            num_classes=num_classes,
        )
