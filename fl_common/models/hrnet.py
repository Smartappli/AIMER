from timm import create_model


def get_hrnet_model(hrnet_type, num_classes):
    """
    Get an HRNet model.

    Parameters:
        hrnet_type (str): Type of HRNet architecture. Options include:
            - "hrnet_w18_small"
            - "hrnet_w18_small_v2"
            - "hrnet_w18"
            - "hrnet_w30"
            - "hrnet_w32"
            - "hrnet_w40"
            - "hrnet_w44"
            - "hrnet_w48"
            - "hrnet_w64"
            - "hrnet_w18_ssld"
            - "hrnet_w48_ssld"
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: HRNet model.

    Raises:
        ValueError: If an unknown HRNet architecture is specified.
    """
    if hrnet_type == 'hrnet_w18_small':
        try:
            hrnet_model = create_model('hrnet_w18_small',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hrnet_model = create_model('hrnet_w18_small',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hrnet_type == 'hrnet_w18_small_v2':
        try:
            hrnet_model = create_model('hrnet_w18_small_v2',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hrnet_model = create_model('hrnet_w18_small_v2',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hrnet_type == 'hrnet_w18':
        try:
            hrnet_model = create_model('hrnet_w18',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hrnet_model = create_model('hrnet_w18',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hrnet_type == 'hrnet_w30':
        try:
            hrnet_model = create_model('hrnet_w30',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hrnet_model = create_model('hrnet_w30',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hrnet_type == 'hrnet_w32':
        try:
            hrnet_model = create_model('hrnet_w32',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hrnet_model = create_model('hrnet_w32',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hrnet_type == 'hrnet_w40':
        try:
            hrnet_model = create_model('hrnet_w40',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hrnet_model = create_model('hrnet_w40',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hrnet_type == 'hrnet_w44':
        try:
            hrnet_model = create_model('hrnet_w44',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hrnet_model = create_model('hrnet_w44',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hrnet_type == 'hrnet_w48':
        try:
            hrnet_model = create_model('hrnet_w48',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hrnet_model = create_model('hrnet_w48',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hrnet_type == 'hrnet_w64':
        try:
            hrnet_model = create_model('hrnet_w64',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hrnet_model = create_model('hrnet_w64',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hrnet_type == 'hrnet_w18_ssld':
        try:
            hrnet_model = create_model('hrnet_w18_ssld',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hrnet_model = create_model('hrnet_w18_ssld',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif hrnet_type == 'hrnet_w48_ssld':
        try:
            hrnet_model = create_model('hrnet_w48_ssld',
                                       pretrained=True,
                                       num_classes=num_classes)
        except Exception:
            hrnet_model = create_model('hrnet_w48_ssld',
                                       pretrained=False,
                                       num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Hrnet Architecture: {hrnet_type}')

    return hrnet_model
