from timm import create_model


def get_hardcorenas_model(hardcorenas_type, num_classes):
    """
    Get a HardcoreNAS (Hardcore Neural Architecture Search) model.

    Parameters:
        hardcorenas_type (str): Type of HardcoreNAS architecture. Options include:
            - 'hardcorenas_a'
            - 'hardcorenas_b'
            - 'hardcorenas_c'
            - 'hardcorenas_d'
            - 'hardcorenas_e'
            - 'hardcorenas_f'
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: HardcoreNAS model.

    Raises:
        ValueError: If an unknown HardcoreNAS architecture is specified.
    """
    if hardcorenas_type == 'hardcorenas_a':
        try:
            hardcorenas_model = create_model('hardcorenas_a',
                                             pretrained=True,
                                             num_classes=num_classes)
        except:
            hardcorenas_model = create_model('hardcorenas_a',
                                             pretrained=False,
                                             num_classes=num_classes)
    elif hardcorenas_type == 'hardcorenas_b':
        try:
            hardcorenas_model = create_model('hardcorenas_b',
                                             pretrained=True,
                                             num_classes=num_classes)
        except:
            hardcorenas_model = create_model('hardcorenas_b',
                                             pretrained=False,
                                             num_classes=num_classes)
    elif hardcorenas_type == 'hardcorenas_c':
        try:
            hardcorenas_model = create_model('hardcorenas_c',
                                             pretrained=True,
                                             num_classes=num_classes)
        except:
            hardcorenas_model = create_model('hardcorenas_c',
                                             pretrained=False,
                                             num_classes=num_classes)
    elif hardcorenas_type == 'hardcorenas_d':
        try:
            hardcorenas_model = create_model('hardcorenas_d',
                                             pretrained=True,
                                             num_classes=num_classes)
        except:
            hardcorenas_model = create_model('hardcorenas_d',
                                             pretrained=False,
                                             num_classes=num_classes)
    elif hardcorenas_type == 'hardcorenas_e':
        try:
            hardcorenas_model = create_model('hardcorenas_e',
                                             pretrained=True,
                                             num_classes=num_classes)
        except:
            hardcorenas_model = create_model('hardcorenas_e',
                                             pretrained=False,
                                             num_classes=num_classes)
    elif hardcorenas_type == 'hardcorenas_f':
        try:
            hardcorenas_model = create_model('hardcorenas_f',
                                             pretrained=True,
                                             num_classes=num_classes)
        except:
            hardcorenas_model = create_model('hardcorenas_f',
                                             pretrained=False,
                                             num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Hardcorenas Architecture: {hardcorenas_type}')

    return hardcorenas_model
