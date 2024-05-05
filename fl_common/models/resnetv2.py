from timm import create_model


def get_resnetv2_model(resnetv2_type, num_classes):
    """
    Load a ResNetV2 model based on the specified type.

    Parameters:
        resnetv2_type (str): The type of ResNetV2 model to load. Options include:
            - 'resnetv2_50x1_bit'
            - 'resnetv2_50x3_bit'
            - 'resnetv2_101x1_bit'
            - 'resnetv2_101x3_bit'
            - 'resnetv2_152x2_bit'
            - 'resnetv2_152x4_bit'
            - 'resnetv2_50'
            - 'resnetv2_50d'
            - 'resnetv2_50t'
            - 'resnetv2_101'
            - 'resnetv2_101d'
            - 'resnetv2_152'
            - 'resnetv2_152d'
            - 'resnetv2_50d_gn'
            - 'resnetv2_50d_evos'
            - 'resnetv2_50d_frn'
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The loaded ResNetV2 model.

    Raises:
        ValueError: If the specified ResNetV2 architecture is unknown.
    """
    if resnetv2_type == 'resnetv2_50x1_bit':
        try:
            resnetv2_model = create_model('resnetv2_50x1_bit',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnetv2_model = create_model('resnetv2_50x1_bit',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnetv2_type == 'resnetv2_50x3_bit':
        try:
            resnetv2_model = create_model('resnetv2_50x3_bit',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnetv2_model = create_model('resnetv2_50x3_bit',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnetv2_type == 'resnetv2_101x1_bit':
        try:
            resnetv2_model = create_model('resnetv2_101x1_bit',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnetv2_model = create_model('resnetv2_101x1_bit',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnetv2_type == 'resnetv2_101x3_bit':
        try:
            resnetv2_model = create_model('resnetv2_101x3_bit',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnetv2_model = create_model('resnetv2_101x3_bit',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnetv2_type == 'resnetv2_152x2_bit':
        try:
            resnetv2_model = create_model('resnetv2_152x2_bit',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnetv2_model = create_model('resnetv2_152x2_bit',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnetv2_type == 'resnetv2_152x4_bit':
        try:
            resnetv2_model = create_model('resnetv2_152x4_bit',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnetv2_model = create_model('resnetv2_152x4_bit',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnetv2_type == 'resnetv2_50':
        try:
            resnetv2_model = create_model('resnetv2_50',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnetv2_model = create_model('resnetv2_50',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnetv2_type == 'resnetv2_50d':
        try:
            resnetv2_model = create_model('resnetv2_50d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnetv2_model = create_model('resnetv2_50d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnetv2_type == 'resnetv2_50t':
        try:
            resnetv2_model = create_model('resnetv2_50t',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnetv2_model = create_model('resnetv2_50t',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnetv2_type == 'resnetv2_101':
        try:
            resnetv2_model = create_model('resnetv2_101',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnetv2_model = create_model('resnetv2_101',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnetv2_type == 'resnetv2_101d':
        try:
            resnetv2_model = create_model('resnetv2_101d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnetv2_model = create_model('resnetv2_101d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnetv2_type == 'resnetv2_152':
        try:
            resnetv2_model = create_model('resnetv2_152',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnetv2_model = create_model('resnetv2_152',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnetv2_type == 'resnetv2_152d':
        try:
            resnetv2_model = create_model('resnetv2_152d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnetv2_model = create_model('resnetv2_152d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnetv2_type == 'resnetv2_50d_gn':
        try:
            resnetv2_model = create_model('resnetv2_50d_gn',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnetv2_model = create_model('resnetv2_50d_gn',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnetv2_type == 'resnetv2_50d_evos':
        try:
            resnetv2_model = create_model('resnetv2_50d_evos',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnetv2_model = create_model('resnetv2_50d_evos',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnetv2_type == 'resnetv2_50d_frn':
        try:
            resnetv2_model = create_model('resnetv2_50d_frn',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnetv2_model = create_model('resnetv2_50d_frn',
                                        pretrained=False,
                                        num_classes=num_classes)
    else:
        raise ValueError(f'Unknown ResNet V2 Architecture: {resnetv2_type}')

    return resnetv2_model
