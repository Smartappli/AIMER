from timm import create_model


def get_hgnet_model(hgnet_type, num_classes):
    """
    Get an HGNet (HourglassNet) model.

    Parameters:
        hgnet_type (str): Type of HGNet architecture. Options include:
            - 'hgnet_tiny'
            - 'hgnet_small'
            - 'hgnet_base'
            - 'hgnetv2_b0'
            - 'hgnetv2_b1'
            - 'hgnetv2_b2'
            - 'hgnetv2_b3'
            - 'hgnetv2_b4'
            - 'hgnetv2_b5'
            - 'hgnetv2_b6'
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: HGNet model.

    Raises:
        ValueError: If an unknown HGNet architecture is specified.
    """
    if hgnet_type == 'hgnet_tiny':
        try:
            hgnet_model = create_model('hgnet_tiny',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hgnet_model = create_model('hgnet_tiny',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hgnet_type == 'hgnet_small':
        try:
            hgnet_model = create_model('hgnet_small',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hgnet_model = create_model('hgnet_small',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hgnet_type == 'hgnet_base':
        try:
            hgnet_model = create_model('hgnet_base',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hgnet_model = create_model('hgnet_base',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hgnet_type == 'hgnetv2_b0':
        try:
            hgnet_model = create_model('hgnetv2_b0',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hgnet_model = create_model('hgnetv2_b0',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hgnet_type == 'hgnetv2_b1':
        try:
            hgnet_model = create_model('hgnetv2_b1',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hgnet_model = create_model('hgnetv2_b1',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hgnet_type == 'hgnetv2_b2':
        try:
            hgnet_model = create_model('hgnetv2_b2',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hgnet_model = create_model('hgnetv2_b2',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hgnet_type == 'hgnetv2_b3':
        try:
            hgnet_model = create_model('hgnetv2_b3',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hgnet_model = create_model('hgnetv2_b3',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hgnet_type == 'hgnetv2_b4':
        try:
            hgnet_model = create_model('hgnetv2_b4',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hgnet_model = create_model('hgnetv2_b4',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hgnet_type == 'hgnetv2_b5':
        try:
            hgnet_model = create_model('hgnetv2_b5',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hgnet_model = create_model('hgnetv2_b5',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hgnet_type == 'hgnetv2_b6':
        try:
            hgnet_model = create_model('hgnetv2_b6',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hgnet_model = create_model('hgnetv2_b6',
                                       pretrained=False,
                                       num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Hgnet Architecture: {hgnet_type}')

    return hgnet_model
