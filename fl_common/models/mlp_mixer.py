from timm import create_model


def get_mlp_mixer_model(mlp_mixer_type, num_classes):
    """
    Create and return an instance of the specified MLP Mixer architecture.

    Args:
    - mlp_mixer_type (str): The type of MLP Mixer architecture to create. It should be one of the following:
                            'mixer_s32_224', 'mixer_s16_224', 'mixer_b32_224', 'mixer_b16_224', 'mixer_l32_224',
                            'mixer_l16_224', 'gmixer_12_224', 'gmixer_24_224', 'resmlp_12_224', 'resmlp_24_224',
                            'resmlp_36_224', 'resmlp_big_24_224', 'gmlp_ti16_224', 'gmlp_s16_224', 'gmlp_b16_224'.
    - num_classes (int): The number of output classes for the model.

    Returns:
    - torch.nn.Module: The created instance of the specified MLP Mixer architecture.

    Raises:
    - ValueError: If an unknown MLP Mixer architecture type is specified.
    """
    if mlp_mixer_type == "mixer_s32_224":
        try:
            mlp_mixer_model = create_model('mixer_s32_224',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            mlp_mixer_model = create_model('mixer_s32_224',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mlp_mixer_type == "mixer_s16_224":
        try:
            mlp_mixer_model = create_model('mixer_s16_224',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            mlp_mixer_model = create_model('mixer_s16_224',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mlp_mixer_type == "mixer_b32_224":
        try:
            mlp_mixer_model = create_model('mixer_b32_224',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            mlp_mixer_model = create_model('mixer_b32_224',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mlp_mixer_type == "mixer_b16_224":
        try:
            mlp_mixer_model = create_model('mixer_b16_224',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            mlp_mixer_model = create_model('mixer_b16_224',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mlp_mixer_type == "mixer_l32_224":
        try:
            mlp_mixer_model = create_model('mixer_l32_224',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            mlp_mixer_model = create_model('mixer_l32_224',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mlp_mixer_type == "mixer_l16_224":
        try:
            mlp_mixer_model = create_model('mixer_l16_224',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            mlp_mixer_model = create_model('mixer_l16_224',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mlp_mixer_type == "gmixer_12_224":
        try:
            mlp_mixer_model = create_model('gmixer_12_224',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            mlp_mixer_model = create_model('gmixer_12_224',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mlp_mixer_type == "gmixer_24_224":
        try:
            mlp_mixer_model = create_model('gmixer_24_224',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            mlp_mixer_model = create_model('gmixer_24_224',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mlp_mixer_type == 'resmlp_12_224':
        try:
            mlp_mixer_model = create_model('resmlp_12_224',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            mlp_mixer_model = create_model('resmlp_12_224',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mlp_mixer_type == 'resmlp_24_224':
        try:
            mlp_mixer_model = create_model('resmlp_24_224',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            mlp_mixer_model = create_model('resmlp_24_224',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mlp_mixer_type == 'resmlp_36_224':
        try:
            mlp_mixer_model = create_model('resmlp_36_224',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            mlp_mixer_model = create_model('resmlp_36_224',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mlp_mixer_type == 'resmlp_big_24_224':
        try:
            mlp_mixer_model = create_model('resmlp_big_24_224',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            mlp_mixer_model = create_model('resmlp_big_24_224',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mlp_mixer_type == 'gmlp_ti16_224':
        try:
            mlp_mixer_model = create_model('gmlp_ti16_224',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            mlp_mixer_model = create_model('gmlp_ti16_224',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mlp_mixer_type == 'gmlp_s16_224':
        try:
            mlp_mixer_model = create_model('gmlp_s16_224',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            mlp_mixer_model = create_model('gmlp_s16_224',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif mlp_mixer_type == 'gmlp_b16_224':
        try:
            mlp_mixer_model = create_model('gmlp_b16_224',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            mlp_mixer_model = create_model('gmlp_b16_224',
                                           pretrained=False,
                                           num_classes=num_classes)
    else:
        # Raise a ValueError if an unknown Mlp Mixer architecture is specified
        raise ValueError(f'Unknown Mlp Mixer Architecture: {mlp_mixer_type}')

    # Return the created Mlp Mixe rmodel
    return mlp_mixer_model
