from timm import create_model


def get_vovnet_model(vovnet_type, num_classes):
    """
    Create and return a Vovnet model based on the specified architecture type and number of classes.

    Parameters:
    - vovnet_type (str): Type of Vovnet architecture. Supported types:
        - 'vovnet39a'
        - 'vovnet57a'
        - 'ese_vovnet19b_slim_dw'
        - 'ese_vovnet19b_slim_dw'
        - 'ese_vovnet19b_slim'
        - 'ese_vovnet39b'
        - 'ese_vovnet57b'
        - 'ese_vovnet99b'
        - 'eca_vovnet39b'
        - 'eca_vovnet39b_evos'
    - num_classes (int): Number of output classes for the model.

    Returns:
    - vovnet_model: A pre-trained Vovnet model with the specified architecture and number of classes.

    Raises:
    - ValueError: If the provided `vovnet_type` is not recognized.
    """
    if vovnet_type == 'vovnet39a':
        try:
            vovnet_model = create_model('vovnet39a',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            vovnet_model = create_model('vovnet39a',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif vovnet_type == 'vovnet57a':
        try:
            vovnet_model = create_model('vovnet57a',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            vovnet_model = create_model('vovnet57a',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif vovnet_type == 'ese_vovnet19b_slim_dw':
        try:
            vovnet_model = create_model('ese_vovnet19b_slim_dw',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            vovnet_model = create_model('ese_vovnet19b_slim_dw',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif vovnet_type == 'ese_vovnet19b_dw':
        try:
            vovnet_model = create_model('ese_vovnet19b_dw',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            vovnet_model = create_model('ese_vovnet19b_dw',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif vovnet_type == 'ese_vovnet19b_slim':
        try:
            vovnet_model = create_model('ese_vovnet19b_slim',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            vovnet_model = create_model('ese_vovnet19b_slim',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif vovnet_type == 'ese_vovnet39b':
        try:
            vovnet_model = create_model('ese_vovnet39b',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            vovnet_model = create_model('ese_vovnet39b',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif vovnet_type == 'ese_vovnet57b':
        try:
            vovnet_model = create_model('ese_vovnet57b',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            vovnet_model = create_model('ese_vovnet57b',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif vovnet_type == 'ese_vovnet99b':
        try:
            vovnet_model = create_model('ese_vovnet99b',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            vovnet_model = create_model('ese_vovnet99b',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif vovnet_type == 'eca_vovnet39b':
        try:
            vovnet_model = create_model('eca_vovnet39b',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            vovnet_model = create_model('eca_vovnet39b',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif vovnet_type == 'ese_vovnet39b_evos':
        try:
            vovnet_model = create_model('ese_vovnet39b_evos',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            vovnet_model = create_model('ese_vovnet39b_evos',
                                        pretrained=False,
                                        num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Vovnet Architecture : {vovnet_type}')

    return vovnet_model
