from timm import create_model


def get_res2net_model(res2net_type, num_classes):
    """
    Returns a Res2Net model based on the provided Res2Net type and number of classes.

    Args:
    - res2net_type (str): The type of Res2Net model. It should be one of the following:
        - 'res2net50_26w_4s'
        - 'res2net101_26w_4s'
        - 'res2net50_26w_6s'
        - 'res2net50_26w_8s'
        - 'res2net50_48w_2s'
        - 'res2net50_14w_8s'
        - 'res2next50'
        - 'res2net50d'
        - 'res2net101d'
    - num_classes (int): The number of output classes for the model.

    Returns:
    - res2net_model: The Res2Net model instantiated based on the specified architecture.

    Raises:
    - ValueError: If the provided res2net_type is not recognized.
    """
    if res2net_type == 'res2net50_26w_4s':
        try:
            res2net_model = create_model('res2net50_26w_4s',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            res2net_model = create_model('res2net50_26w_4s',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif res2net_type == 'res2net101_26w_4s':
        try:
            res2net_model = create_model('res2net101_26w_4s',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            res2net_model = create_model('res2net101_26w_4s',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif res2net_type == 'res2net50_26w_6s':
        try:
            res2net_model = create_model('res2net50_26w_6s',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            res2net_model = create_model('res2net50_26w_6s',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif res2net_type == 'res2net50_26w_8s':
        try:
            res2net_model = create_model('res2net50_26w_8s',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            res2net_model = create_model('res2net50_26w_8s',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif res2net_type == 'res2net50_48w_2s':
        try:
            res2net_model = create_model('res2net50_48w_2s',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            res2net_model = create_model('res2net50_48w_2s',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif res2net_type == 'res2net50_14w_8s':
        try:
            res2net_model = create_model('res2net50_14w_8s',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            res2net_model = create_model('res2net50_14w_8s',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif res2net_type == 'res2next50':
        try:
            res2net_model = create_model('res2next50',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            res2net_model = create_model('res2next50',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif res2net_type == 'res2net50d':
        try:
            res2net_model = create_model('res2net50d',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            res2net_model = create_model('res2net50d',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif res2net_type == 'res2net101d':
        try:
            res2net_model = create_model('res2net101d',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            res2net_model = create_model('res2net101d',
                                         pretrained=False,
                                         num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Res2net Architecture: {res2net_type}')

    return res2net_model
