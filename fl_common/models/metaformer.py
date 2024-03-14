from timm import create_model


def get_metaformer_model(metaformer_type, num_classes):
    """
    Get a Metaformer model based on the specified architecture type.

    Args:
        metaformer_type (str): The type of Metaformer architecture. It can be one of the following:
            - 'poolformer_s12': Poolformer with size 12.
            - 'poolformer_s24': Poolformer with size 24.
            - 'poolformer_s36': Poolformer with size 36.
            - 'poolformer_m36': Poolformer with medium size 36.
            - 'poolformer_m48': Poolformer with medium size 48.
            - 'poolformerv2_s12': PoolformerV2 with size 12.
            - 'poolformerv2_s24': PoolformerV2 with size 24.
            - 'poolformerv2_s36': PoolformerV2 with size 36.
            - 'poolformerv2_m36': PoolformerV2 with medium size 36.
            - 'poolformerv2_m48': PoolformerV2 with medium size 48.
            - 'convformer_s18': Convformer with small size 18.
            - 'convformer_s36': Convformer with small size 36.
            - 'convformer_m36': Convformer with medium size 36.
            - 'convformer_b36': Convformer with big size 36.
            - 'caformer_s18': CAformer with small size 18.
            - 'caformer_s36': CAformer with small size 36.
            - 'caformer_m36': CAformer with medium size 36.
            - 'caformer_b36': CAformer with big size 36.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The Metaformer model.

    Raises:
        ValueError: If an unknown Metaformer architecture type is specified.
    """
    if metaformer_type == 'poolformer_s12':
        try:
            metaformer_model = create_model('poolformer_s12',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('poolformer_s12',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'poolformer_s24':
        try:
            metaformer_model = create_model('poolformer_s24',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('poolformer_s24',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'poolformer_s36':
        try:
            metaformer_model = create_model('poolformer_s36',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('poolformer_s36',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'poolformer_m36':
        try:
            metaformer_model = create_model('poolformer_m36',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('poolformer_m36',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'poolformer_m48':
        try:
            metaformer_model = create_model('poolformer_m48',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('poolformer_m48',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'poolformerv2_s12':
        try:
            metaformer_model = create_model('poolformerv2_s12',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('poolformerv2_s12',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'poolformerv2_s24':
        try:
            metaformer_model = create_model('poolformerv2_s24',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('poolformerv2_s24',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'poolformerv2_s36':
        try:
            metaformer_model = create_model('poolformerv2_s36',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('poolformerv2_s36',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'poolformerv2_m36':
        try:
            metaformer_model = create_model('poolformerv2_m36',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('poolformerv2_m36',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'poolformerv2_m48':
        try:
            metaformer_model = create_model('poolformerv2_m48',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('poolformerv2_m48',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'convformer_s18':
        try:
            metaformer_model = create_model('convformer_s18',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('convformer_s18',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'convformer_s36':
        try:
            metaformer_model = create_model('convformer_s36',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('convformer_s36',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'convformer_m36':
        try:
            metaformer_model = create_model('convformer_m36',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('convformer_m36',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'convformer_b36':
        try:
            metaformer_model = create_model('convformer_b36',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('convformer_b36',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'caformer_s18':
        try:
            metaformer_model = create_model('caformer_s18',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('caformer_s18',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'caformer_s36':
        try:
            metaformer_model = create_model('caformer_s36',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('caformer_s36',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'caformer_m36':
        try:
            metaformer_model = create_model('caformer_m36',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('caformer_m36',
                                            pretrained=False,
                                            num_classes=num_classes)
    elif metaformer_type == 'caformer_b36':
        try:
            metaformer_model = create_model('caformer_b36',
                                            pretrained=True,
                                            num_classes=num_classes)
        except:
            metaformer_model = create_model('caformer_b36',
                                            pretrained=False,
                                            num_classes=num_classes)
    else:
        # Raise a ValueError if an unknown Metaformer architecture is specified
        raise ValueError(f'Unknown Metaformer Architecture: {metaformer_type}')

    # Return the created Metaformer model
    return metaformer_model
