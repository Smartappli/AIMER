from timm import create_model


def get_efficientvit_msra_model(efficientvit_msra_type, num_classes):
    """
    Get an EfficientViT-MSRA model of specified type.

    Args:
        efficientvit_msra_type (str): Type of EfficientViT-MSRA model.
            It should be one of ['efficientvit_m0', 'efficientvit_m1', 'efficientvit_m2', 'efficientvit_m3',
            'efficientvit_m4', 'efficientvit_m5'].
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: EfficientViT-MSRA model.
    """
    if efficientvit_msra_type == 'efficientvit_m0':
        try:
            efficientvit_msra_model = create_model('efficientvit_m0',
                                                 pretrained=True,
                                                 num_classes=num_classes)
        except:
            efficientvit_msra_model = create_model('efficientvit_m0',
                                                 pretrained=False,
                                                 num_classes=num_classes)
    elif efficientvit_msra_type == 'efficientvit_m1':
        try:
            efficientvit_msra_model = create_model('efficientvit_m1',
                                                 pretrained=True,
                                                 num_classes=num_classes)
        except:
            efficientvit_msra_model = create_model('efficientvit_m1',
                                                 pretrained=False,
                                                 num_classes=num_classes)
    elif efficientvit_msra_type == 'efficientvit_m2':
        try:
            efficientvit_msra_model = create_model('efficientvit_m2',
                                                 pretrained=True,
                                                 num_classes=num_classes)
        except:
            efficientvit_msra_model = create_model('efficientvit_m2',
                                                 pretrained=False,
                                                 num_classes=num_classes)
    elif efficientvit_msra_type == 'efficientvit_m3':
        try:
            efficientvit_msra_model = create_model('efficientvit_m3',
                                                 pretrained=True,
                                                 num_classes=num_classes)
        except:
            efficientvit_msra_model = create_model('efficientvit_m3',
                                                 pretrained=False,
                                                 num_classes=num_classes)
    elif efficientvit_msra_type == 'efficientvit_m4':
        try:
            efficientvit_msra_model = create_model('efficientvit_m4',
                                                 pretrained=True,
                                                 num_classes=num_classes)
        except:
            efficientvit_msra_model = create_model('efficientvit_m4',
                                                 pretrained=False,
                                                 num_classes=num_classes)
    elif efficientvit_msra_type == 'efficientvit_m5':
        try:
            efficientvit_msra_model = create_model('efficientvit_m5',
                                                 pretrained=True,
                                                 num_classes=num_classes)
        except:
            efficientvit_msra_model = create_model('efficientvit_m5',
                                                 pretrained=False,
                                                 num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Efficientvit_msra Architecture: {efficientvit_msra_type}')

    return efficientvit_msra_model
