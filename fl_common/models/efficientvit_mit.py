from timm import create_model


def get_efficientvit_mit_model(efficientvit_mit_type, num_classes):
    """
    Returns an EfficientViT model based on the specified type and number of classes.

    Args:
        efficientvit_mit_type (str): Type of EfficientViT model. Supported types are
            'efficientvit_b0', 'efficientvit_b1', 'efficientvit_b2', 'efficientvit_b3',
            'efficientvit_l1', 'efficientvit_l2', 'efficientvit_l3'.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: EfficientViT model instance.

    Raises:
        ValueError: If the specified EfficientViT model type is not recognized.
    """
    if efficientvit_mit_type == 'efficientvit_b0':
        try:
            efficientvit_mit_model = create_model('efficientvit_b0',
                                                  pretrained=True,
                                                  num_classes=num_classes)
        except RuntimeError:
            efficientvit_mit_model = create_model('efficientvit_b0',
                                                  pretrained=False,
                                                  num_classes=num_classes)
    elif efficientvit_mit_type == 'efficientvit_b1':
        try:
            efficientvit_mit_model = create_model('efficientvit_b1',
                                                  pretrained=True,
                                                  num_classes=num_classes)
        except RuntimeError:
            efficientvit_mit_model = create_model('efficientvit_b1',
                                                  pretrained=False,
                                                  num_classes=num_classes)
    elif efficientvit_mit_type == 'efficientvit_b2':
        try:
            efficientvit_mit_model = create_model('efficientvit_b2',
                                                  pretrained=True,
                                                  num_classes=num_classes)
        except RuntimeError:
            efficientvit_mit_model = create_model('efficientvit_b2',
                                                  pretrained=False,
                                                  num_classes=num_classes)
    elif efficientvit_mit_type == 'efficientvit_b3':
        try:
            efficientvit_mit_model = create_model('efficientvit_b3',
                                                  pretrained=True,
                                                  num_classes=num_classes)
        except RuntimeError:
            efficientvit_mit_model = create_model('efficientvit_b3',
                                                  pretrained=False,
                                                  num_classes=num_classes)
    elif efficientvit_mit_type == 'efficientvit_l1':
        try:
            efficientvit_mit_model = create_model('efficientvit_l1',
                                                  pretrained=True,
                                                  num_classes=num_classes)
        except RuntimeError:
            efficientvit_mit_model = create_model('efficientvit_l1',
                                                  pretrained=False,
                                                  num_classes=num_classes)
    elif efficientvit_mit_type == 'efficientvit_l2':
        try:
            efficientvit_mit_model = create_model('efficientvit_l2',
                                                  pretrained=True,
                                                  num_classes=num_classes)
        except RuntimeError:
            efficientvit_mit_model = create_model('efficientvit_l2',
                                                  pretrained=False,
                                                  num_classes=num_classes)
    elif efficientvit_mit_type == 'efficientvit_l3':
        try:
            efficientvit_mit_model = create_model('efficientvit_l3',
                                                  pretrained=True,
                                                  num_classes=num_classes)
        except RuntimeError:
            efficientvit_mit_model = create_model('efficientvit_l3',
                                                  pretrained=False,
                                                  num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Efficientvit_mit Architecture: {efficientvit_mit_type}')

    return efficientvit_mit_model
