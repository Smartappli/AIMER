from timm import create_model


def get_coat_model(coat_type, num_classes):
    """
    Create and return a COAT model based on the specified architecture.

    Parameters:
    - coat_type (str): Type of COAT architecture ('coat_tiny', 'coat_mini', 'coat_small',
                      'coat_lite_tiny', 'coat_lite_mini', 'coat_lite_small', 'coat_lite_medium',
                      'coat_lite_medium_384').
    - num_classes (int): Number of output classes.

    Returns:
    - coat_model: Created COAT model.

    Raises:
    - ValueError: If an unknown COAT architecture is specified.
    """
    if coat_type == 'coat_tiny':
        try:
            coat_model = create_model('coat_tiny', pretrained=True, num_classes=num_classes)
        except:
            coat_model = create_model('coat_tiny', pretrained=False, num_classes=num_classes)
    elif coat_type == 'coat_mini':
        try:
            coat_model = create_model('coat_mini', pretrained=True, num_classes=num_classes)
        except:
            coat_model = create_model('coat_mini', pretrained=False, num_classes=num_classes)
    elif coat_type == 'coat_small':
        try:
            coat_model = create_model('coat_small', pretrained=True, num_classes=num_classes)
        except:
            coat_model = create_model('coat_small', pretrained=False, num_classes=num_classes)
    elif coat_type == 'coat_lite_tiny':
        try:
            coat_model = create_model('coat_lite_tiny', pretrained=True, num_classes=num_classes)
        except:
            coat_model = create_model('coat_lite_tiny', pretrained=False, num_classes=num_classes)
    elif coat_type == 'coat_lite_mini':
        try:
            coat_model = create_model('coat_lite_mini', pretrained=True, num_classes=num_classes)
        except:
            coat_model = create_model('coat_lite_mini', pretrained=False, num_classes=num_classes)
    elif coat_type == 'coat_lite_small':
        try:
            coat_model = create_model('coat_lite_small', pretrained=True, num_classes=num_classes)
        except:
            coat_model = create_model('coat_lite_small', pretrained=False, num_classes=num_classes)
    elif coat_type == 'coat_lite_medium':
        try:
            coat_model = create_model('coat_lite_medium', pretrained=True, num_classes=num_classes)
        except:
            coat_model = create_model('coat_lite_medium', pretrained=False, num_classes=num_classes)
    elif coat_type == 'coat_lite_medium_384':
        try:
            coat_model = create_model('coat_lite_medium', pretrained=True, num_classes=num_classes)
        except:
            coat_model = create_model('coat_lite_medium', pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Cait Architecture: {coat_type}')

    return coat_model
