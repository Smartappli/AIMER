from timm import create_model


def get_twins_model(twins_type, num_classes):
    """
    Get a Twins model.

    Parameters:
        twins_type (str): Type of Twins architecture. Options include:
            - "twins_pcpvt_small"
            - "twins_pcpvt_base"
            - "twins_pcpvt_large"
            - "twins_svt_small"
            - "twins_svt_base"
            - "twins_svt_large"
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Twins model.

    Raises:
        ValueError: If an unknown Twins architecture is specified.
    """
    if twins_type == 'twins_pcpvt_small':
        try:
            twins_model = create_model('twins_pcpvt_small',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            twins_model = create_model('twins_pcpvt_small',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif twins_type == 'twins_pcpvt_base':
        try:
            twins_model = create_model('twins_pcpvt_base',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            twins_model = create_model('twins_pcpvt_base',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif twins_type == 'twins_pcpvt_large':
        try:
            twins_model = create_model('twins_pcpvt_large',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            twins_model = create_model('twins_pcpvt_large',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif twins_type == 'twins_svt_small':
        try:
            twins_model = create_model('twins_svt_small',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            twins_model = create_model('twins_svt_small',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif twins_type == 'twins_svt_base':
        try:
            twins_model = create_model('twins_svt_base',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            twins_model = create_model('twins_svt_base',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif twins_type == 'twins_svt_large':
        try:
            twins_model = create_model('twins_svt_large',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            twins_model = create_model('twins_svt_large',
                                       pretrained=False,
                                       num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Twins Architecture: {twins_type}')

    return twins_model
