from timm import create_model


def get_senet_model(senet_type, num_classes):
    """
    Get a SENet model based on the specified architecture type.

    Args:
        senet_type (str): The type of SENet architecture. It can be one of the following:
            - 'legacy_seresnet18': Legacy SE-ResNet18 architecture.
            - 'legacy_seresnet34': Legacy SE-ResNet34 architecture.
            - 'legacy_seresnet50': Legacy SE-ResNet50 architecture.
            - 'legacy_seresnet101': Legacy SE-ResNet101 architecture.
            - 'legacy_seresnet152': Legacy SE-ResNet152 architecture.
            - 'legacy_senet154': Legacy SE-Net154 architecture.
            - 'legacy_seresnext26_32x4d': Legacy SE-ResNeXt26_32x4d architecture.
            - 'legacy_seresnext50_32x4d': Legacy SE-ResNeXt50_32x4d architecture.
            - 'legacy_seresnext101_32x4d': Legacy SE-ResNeXt101_32x4d architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The SENet model.

    Raises:
        ValueError: If an unknown SENet architecture type is specified.
    """
    if senet_type == 'legacy_seresnet18':
        try:
            senet_model = create_model('legacy_seresnet18',
                                       pretrained=True,
                                       num_classes=num_classes)
        except:
            senet_model = create_model('legacy_seresnet18',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif senet_type == 'legacy_seresnet34':
        try:
            senet_model = create_model('legacy_seresnet34',
                                       pretrained=True,
                                       num_classes=num_classes)
        except:
            senet_model = create_model('legacy_seresnet34',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif senet_type == 'legacy_seresnet50':
        try:
            senet_model = create_model('legacy_seresnet50',
                                       pretrained=True,
                                       num_classes=num_classes)
        except:
            senet_model = create_model('legacy_seresnet34',
                                       pretrained=True,
                                       num_classes=num_classes)
    elif senet_type == 'legacy_seresnet101':
        try:
            senet_model = create_model('legacy_seresnet101',
                                       pretrained=True,
                                       num_classes=num_classes)
        except:
            senet_model = create_model('legacy_seresnet101',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif senet_type == 'legacy_seresnet152':
        try:
            senet_model = create_model('legacy_seresnet152',
                                       pretrained=True,
                                       num_classes=num_classes)
        except:
            senet_model = create_model('legacy_seresnet152',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif senet_type == 'legacy_senet154':
        try:
            senet_model = create_model('legacy_senet154',
                                       pretrained=True,
                                       num_classes=num_classes)
        except:
            senet_model = create_model('legacy_senet154',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif senet_type == 'legacy_seresnext26_32x4d':
        try:
            senet_model = create_model('legacy_seresnext26_32x4d',
                                       pretrained=True,
                                       num_classes=num_classes)
        except:
            senet_model = create_model('legacy_seresnext26_32x4d',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif senet_type == 'legacy_seresnext50_32x4d':
        try:
            senet_model = create_model('legacy_seresnext50_32x4d',
                                       pretrained=True,
                                       num_classes=num_classes)
        except:
            senet_model = create_model('legacy_seresnext50_32x4d',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif senet_type == 'legacy_seresnext101_32x4d':
        try:
            senet_model = create_model('legacy_seresnext101_32x4d',
                                       pretrained=True,
                                       num_classes=num_classes)
        except:
            senet_model = create_model('legacy_seresnext101_32x4d',
                                       pretrained=False,
                                       num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Senet Architecture: {senet_type}')

    return senet_model
