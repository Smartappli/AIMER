from timm import create_model


def get_convit_model(convit_type, num_classes):
    """
    Get a Convit model based on the specified architecture.

    Parameters:
        convit_type (str): Type of Convit architecture to be used.
                          Supported values: "convit_tiny", "convit_small", "convit_base".
        num_classes (int): Number of output classes for the model.

    Returns:
        torch.nn.Module: Convit model instantiated based on the specified architecture.

    Raises:
        ValueError: If the provided convit_type is not recognized.

    Example:
        To get a Convit Tiny model with 10 output classes:
        >>> model = get_convit_model("convit_tiny", num_classes=10)
    """
    if convit_type == "convit_tiny":
        convit_model = create_model('convit_tiny', pretrained=True, num_classes=num_classes)
    elif convit_type == "convit_small":
        convit_model = create_model('convit_small', pretrained=True, num_classes=num_classes)
    elif convit_type == "convit_base":
        convit_model = create_model('convit_base', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Convit Architecture: {convit_type}')

    return convit_model
