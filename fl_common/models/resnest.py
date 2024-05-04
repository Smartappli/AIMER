from timm import create_model


def get_resnest_model(resnest_type, num_classes):
    """
    Create a ResNeSt model of specified type and number of classes.

    Args:
        resnest_type (str): Type of ResNeSt model to create. Supported types are:
            - 'resnest14d'
            - 'resnest26d'
            - 'resnest50d'
            - 'resnest101e'
            - 'resnest200e'
            - 'resnest269e'
            - 'resnest50d_4s2x40d'
            - 'resnest50d_1s4x24d'
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: ResNeSt model with specified type and number of classes.

    Raises:
        ValueError: If an unknown Resnest architecture is provided.
    """
    if resnest_type == 'resnest14d':
        try:
            resnest_model = create_model('resnest14d',
                                         pretrained=True,
                                         num_classes=num_classes)
        except:
            resnest_model = create_model('resnest14d',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif resnest_type == 'resnest26d':
        try:
            resnest_model = create_model('resnest26d',
                                         pretrained=True,
                                         num_classes=num_classes)
        except:
            resnest_model = create_model('resnest26d',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif resnest_type == 'resnest50d':
        try:
            resnest_model = create_model('resnest50d',
                                         pretrained=True,
                                         num_classes=num_classes)
        except:
            resnest_model = create_model('resnest50d',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif resnest_type == 'resnest101e':
        try:
            resnest_model = create_model('resnest101e',
                                         pretrained=True,
                                         num_classes=num_classes)
        except:
            resnest_model = create_model('resnest101e',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif resnest_type == 'resnest200e':
        try:
            resnest_model = create_model('resnest200e',
                                         pretrained=True,
                                         num_classes=num_classes)
        except:
            resnest_model = create_model('resnest200e',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif resnest_type == 'resnest269e':
        try:
            resnest_model = create_model('resnest269e',
                                         pretrained=True,
                                         num_classes=num_classes)
        except:
            resnest_model = create_model('resnest269e',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif resnest_type == 'resnest50d_4s2x40d':
        try:
            resnest_model = create_model('resnest50d_4s2x40d',
                                         pretrained=True,
                                         num_classes=num_classes)
        except:
            resnest_model = create_model('resnest50d_4s2x40d',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif resnest_type == 'resnest50d_1s4x24d':
        try:
            resnest_model = create_model('resnest50d_1s4x24d',
                                         pretrained=True,
                                         num_classes=num_classes)
        except:
            resnest_model = create_model('resnest50d_1s4x24d',
                                         pretrained=False,
                                         num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Resnest Architecture: {resnest_type}')

    return resnest_model
