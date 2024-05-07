from timm import create_model


def get_convit_model(convit_type, num_classes):
    """
    Get a Convit model based on the specified architecture.

    Args:
        convit_type (str): Type of Convit architecture to be used.
        num_classes (int): Number of output classes for the model.

    Returns:
        torch.nn.Module: Convit model instantiated based on the specified architecture.

    Raises:
        ValueError: If the provided convit_type is not recognized.
    """
    valid_convit_types = {'convit_tiny', 'convit_small', 'convit_base'}

    if convit_type not in valid_convit_types:
        raise ValueError(f'Unknown Convit Architecture: {convit_type}')

    try:
        return create_model(convit_type, pretrained=True, num_classes=num_classes)
    except Exception:
        return create_model(convit_type, pretrained=False, num_classes=num_classes)
