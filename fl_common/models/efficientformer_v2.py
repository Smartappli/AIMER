from timm import create_model


def get_efficientformer_v2_model(efficientformer_v2_type, num_classes):
    """
    Get an Efficientformer v2 model of specified type.

    Args:
        efficientformer_v2_type (str): Type of Efficientformer v2 model.
            It should be one of ['efficientformerv2_s0', 'efficientformerv2_s1', 'efficientformerv2_s2', 'efficientformerv2_l'].
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Efficientformer v2 model.
    """
    if efficientformer_v2_type == 'efficientformerv2_s0':
        try:
            efficientformer_v2_model = create_model('efficientformerv2_s0',
                                                    pretrained=True,
                                                    num_classes=num_classes)
        except OSError:
            efficientformer_v2_model = create_model('efficientformerv2_s0',
                                                    pretrained=False,
                                                    num_classes=num_classes)
    elif efficientformer_v2_type == 'efficientformerv2_s1':
        try:
            efficientformer_v2_model = create_model('efficientformerv2_s1',
                                                    pretrained=True,
                                                    num_classes=num_classes)
        except OSError:
            efficientformer_v2_model = create_model('efficientformerv2_s1',
                                                    pretrained=False,
                                                    num_classes=num_classes)
    elif efficientformer_v2_type == 'efficientformerv2_s2':
        try:
            efficientformer_v2_model = create_model('efficientformerv2_s2',
                                                    pretrained=True,
                                                    num_classes=num_classes)
        except OSError:
            efficientformer_v2_model = create_model('efficientformerv2_s2',
                                                    pretrained=False,
                                                    num_classes=num_classes)
    elif efficientformer_v2_type == 'efficientformerv2_l':
        try:
            efficientformer_v2_model = create_model('efficientformerv2_l',
                                                    pretrained=True,
                                                    num_classes=num_classes)
        except OSError:
            efficientformer_v2_model = create_model('efficientformerv2_l',
                                                    pretrained=False,
                                                    num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Efficientformer v2 Architecture: {efficientformer_v2_type}')

    return efficientformer_v2_model
