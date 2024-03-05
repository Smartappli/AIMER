from timm import create_model


def get_selecsls_model(selecsls_type, num_classes):
    """
    Get a SelecSLS model based on the specified architecture type.

    Args:
        selecsls_type (str): The type of SelecSLS architecture. It can be one of the following:
            - 'selecsls42': SelecSLS-42.
            - 'selecsls42b': SelecSLS-42B.
            - 'selecsls60': SelecSLS-60.
            - 'selecsls60b': SelecSLS-60B.
            - 'selecsls84': SelecSLS-84.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The SelecSLS model.

    Raises:
        ValueError: If an unknown SelecSLS architecture type is specified.
    """
    if selecsls_type == 'selecsls42':
        try:
            selecsls_model = create_model('selecsls42',
                                          pretrained=True,
                                          num_classes=num_classes)
        except:
            selecsls_model = create_model('selecsls42',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif selecsls_type == 'selecsls42b':
        try:
            selecsls_model = create_model('selecsls42b',
                                          pretrained=True,
                                          num_classes=num_classes)
        except:
            selecsls_model = create_model('selecsls42b',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif selecsls_type == 'selecsls60':
        try:
            selecsls_model = create_model('selecsls60',
                                          pretrained=True,
                                          num_classes=num_classes)
        except:
            selecsls_model = create_model('selecsls60',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif selecsls_type == 'selecsls60b':
        try:
            selecsls_model = create_model('selecsls60b',
                                          pretrained=True,
                                          num_classes=num_classes)
        except:
            selecsls_model = create_model('selecsls60b',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif selecsls_type == 'selecsls84':
        try:
            selecsls_model = create_model('selecsls84',
                                          pretrained=True,
                                          num_classes=num_classes)
        except:
            selecsls_model = create_model('selecsls84',
                                          pretrained=False,
                                          num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Selecsls Architecture: {selecsls_type}')

    return selecsls_model
