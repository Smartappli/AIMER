from timm import create_model


def get_coat_model(coat_type, num_classes):
    """
    Create and return a COAT model based on the specified architecture.

    Args:
        coat_type (str): Type of COAT architecture.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Created COAT model.

    Raises:
        ValueError: If an unknown COAT architecture is specified.
    """
    valid_coat_types = {
        'coat_tiny', 'coat_mini', 'coat_small',
        'coat_lite_tiny', 'coat_lite_mini', 'coat_lite_small',
        'coat_lite_medium', 'coat_lite_medium_384'
    }

    if coat_type not in valid_coat_types:
        raise ValueError(f'Unknown COAT Architecture: {coat_type}')

    try:
        return create_model(coat_type, pretrained=True, num_classes=num_classes)
    except OSError as e:
        print(f"Error loading pretrained model: {e}")
        return create_model(coat_type, pretrained=False, num_classes=num_classes)
