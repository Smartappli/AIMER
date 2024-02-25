from timm import create_model


def get_tresnet_model(tresnet_type, num_classes):
    """
    Get a TResNet model.

    Parameters:
        tresnet_type (str): Type of TResNet architecture. Options include:
            - "tresnet_m"
            - "tresnet_l"
            - "tresnet_xl"
            - "tresnet_v2_l"
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: TResNet model.

    Raises:
        ValueError: If an unknown TResNet architecture is specified.
    """
    if tresnet_type == 'tresnet_m':
        tresnet_model = create_model('tresnet_m', pretrained=True, num_classes=num_classes)
    elif tresnet_type == 'tresnet_l':
        tresnet_model = create_model('tresnet_l', pretrained=True, num_classes=num_classes)
    elif tresnet_type == 'tresnet_xl':
        tresnet_model = create_model('tresnet_xl', pretrained=True, num_classes=num_classes)
    elif tresnet_type == 'tresnet_v2_l':
        tresnet_model = create_model('tresnet_v2_l', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Tresnet Architecture: {tresnet_type}')

    return tresnet_model
