from timm import create_model


def get_davit_model(davit_type, num_classes):
    """
    Create and return a Davit model based on the specified architecture.

    Parameters:
    - davit_type (str): Type of Davit architecture.
    - num_classes (int): Number of output classes.

    Returns:
    - davit_model: Created Davit model.
    """

    # List of valid Davit architectures
    valid_davit_types = [
        "davit_tiny",
        "davit_small",
        "davit_base",
        "davit_large",
        "davit_huge",
        "davit_giant",
    ]

    # Check if the davit_type is valid
    if davit_type not in valid_davit_types:
        msg = f"Unknown Davit Architecture: {davit_type}"
        raise ValueError(msg)

    try:
        return create_model(davit_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{davit_type} - Error loading pretrained model: {e}")
        return create_model(davit_type, pretrained=False, num_classes=num_classes)
