from timm import create_model


def get_twins_model(twins_type, num_classes):
    """
    Get a Twins model.

    Parameters:
        twins_type (str): Type of Twins architecture.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Twins model.

    Raises:
        ValueError: If an unknown Twins architecture is specified.
    """
    valid_types = {
        "twins_pcpvt_small",
        "twins_pcpvt_base",
        "twins_pcpvt_large",
        "twins_svt_small",
        "twins_svt_base",
        "twins_svt_large",
    }

    if twins_type not in valid_types:
        msg = f"Unknown Twins Architecture: {twins_type}"
        raise ValueError(msg)

    try:
        return create_model(
            twins_type, pretrained=True, num_classes=num_classes,
        )
    except RuntimeError as e:
        print(f"{twins_type} - Error loading pretrained model: {e}")
        return create_model(
            twins_type, pretrained=False, num_classes=num_classes,
        )
