from timm import create_model


def get_xception_model(xception_type, num_classes):
    """
    Load a pre-trained Xception model of the specified type and modify its
    last layer to accommodate the given number of classes.

    Parameters:
    - xception_type (str): Type of Xception architecture, supported types:
        - 'legacy_xception'
        - 'xception41'
        - 'xception65'
        - 'xception71'
        - 'xception41p'
        - 'xception65p'
    - num_classes (int): Number of output classes for the modified last layer.

    Returns:
    - torch.nn.Module: Modified Xception model with the specified architecture
      and last layer adapted for the given number of classes.
    """
    if xception_type == 'legacy_xception':
        try:
            xception_model = create_model('legacy_xception',
                                          pretrained=True,
                                          num_classes=num_classes)
        except OSError:
            xception_model = create_model('legacy_xception',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif xception_type == 'xception41':
        try:
            xception_model = create_model('xception41',
                                          pretrained=True,
                                          num_classes=num_classes)
        except OSError:
            xception_model = create_model('xception41',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif xception_type == 'xception65':
        try:
            xception_model = create_model('xception65',
                                          pretrained=True,
                                          num_classes=num_classes)
        except OSError:
            xception_model = create_model('xception65',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif xception_type == 'xception71':
        try:
            xception_model = create_model('xception71',
                                          pretrained=True,
                                          num_classes=num_classes)
        except OSError:
            xception_model = create_model('xception71',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif xception_type == 'xception41p':
        try:
            xception_model = create_model('xception41',
                                          pretrained=True,
                                          num_classes=num_classes)
        except OSError:
            xception_model = create_model('xception41',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif xception_type == 'xception65p':
        try:
            xception_model = create_model('xception65p',
                                          pretrained=True,
                                          num_classes=num_classes)
        except OSError:
            xception_model = create_model('xception65p',
                                          pretrained=False,
                                          num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Xception Architecture: {xception_type}')

    return xception_model
