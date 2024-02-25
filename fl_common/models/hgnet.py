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
        hgnet_model = create_model('hgnet_tiny', pretrained=True, num_classes=num_classes)
    elif hgnet_type == 'hgnet_small':
        hgnet_model = create_model('hgnet_small', pretrained=True, num_classes=num_classes)
    elif hgnet_type == 'hgnet_base':
        hgnet_model = create_model('hgnet_base', pretrained=True, num_classes=num_classes)
    elif hgnet_type == 'hgnetv2_b0':
        hgnet_model = create_model('hgnetv2_b0', pretrained=True, num_classes=num_classes)
    elif hgnet_type == 'hgnetv2_b1':
        hgnet_model = create_model('hgnetv2_b1', pretrained=True, num_classes=num_classes)
    elif hgnet_type == 'hgnetv2_b2':
        hgnet_model = create_model('hgnetv2_b2', pretrained=True, num_classes=num_classes)
    elif hgnet_type == 'hgnetv2_b3':
        hgnet_model = create_model('hgnetv2_b3', pretrained=True, num_classes=num_classes)
    elif hgnet_type == 'hgnetv2_b4':
        hgnet_model = create_model('hgnetv2_b4', pretrained=True, num_classes=num_classes)
    elif hgnet_type == 'hgnetv2_b5':
        hgnet_model = create_model('hgnetv2_b5', pretrained=True, num_classes=num_classes)
    elif hgnet_type == 'hgnetv2_b6':
        hgnet_model = create_model('hgnetv2_b6', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Hgnet Architecture: {hgnet_type}')

    return hgnet_model
