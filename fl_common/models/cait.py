from timm import create_model


def get_cait_model(cait_type, num_classes):
    """
    Create and return a CAIT model based on the specified architecture.

    Args:
        cait_type (str): Type of CAIT architecture.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Created CAIT model.

    Raises:
        ValueError: If an unknown CAIT architecture is specified.
    """
    valid_cait_types = {
        "cait_xxs24_224",
        "cait_xxs24_384",
        "cait_xxs36_224",
        "cait_xxs36_384",
        "cait_xs24_384",
        "cait_s24_224",
        "cait_s24_384",
        "cait_s36_384",
        "cait_m36_224",
        "cait_m36_384",
        "cait_m48_448",
    }

    if cait_type not in valid_cait_types:
        raise ValueError(f"Unknown CAIT Architecture: {cait_type}")

    try:
        return create_model(
            cait_type,
            pretrained=True,
            num_classes=num_classes)
    except RuntimeError as e:
        print(f"{cait_type} - Error loading pretrained model: {e}")
        return create_model(
            cait_type,
            pretrained=False,
            num_classes=num_classes)
