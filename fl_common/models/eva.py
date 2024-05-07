from timm import create_model


def get_eva_model(eva_type, num_classes):
    """
    Get an EVA (Efficient Vision Architecture) model.

    Parameters:
        eva_type (str): Type of EVA architecture. Options include:
            - "eva_giant_patch14_224"
            - "eva_giant_patch14_336"
            - "eva_giant_patch14_560"
            - "eva02_tiny_patch14_224"
            - "eva02_small_patch14_224"
            - "eva02_base_patch14_224"
            - "eva02_large_patch14_224"
            - "eva02_tiny_patch14_336"
            - "eva02_small_patch14_336"
            - "eva02_base_patch14_448"
            - "eva02_large_patch14_448"
            - "eva_giant_patch14_clip_224"
            - "eva02_base_patch16_clip_224"
            - "eva02_large_patch14_clip_224"
            - "eva02_large_patch14_clip_336"
            - "eva02_enormous_patch14_clip_224"
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: EVA model.

    Raises:
        ValueError: If an unknown Eva architecture is specified.
    """
    if eva_type == 'eva_giant_patch14_224':
        try:
            eva_model = create_model('eva_giant_patch14_224',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            eva_model = create_model('eva_giant_patch14_224',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif eva_type == 'eva_giant_patch14_336':
        try:
            eva_model = create_model('eva_giant_patch14_336',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            eva_model = create_model('eva_giant_patch14_336',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif eva_type == 'eva_giant_patch14_560':
        try:
            eva_model = create_model('eva_giant_patch14_560',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            eva_model = create_model('eva_giant_patch14_560',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif eva_type == 'eva02_tiny_patch14_224':
        try:
            eva_model = create_model('eva02_tiny_patch14_224',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            eva_model = create_model('eva02_tiny_patch14_224',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif eva_type == 'eva02_small_patch14_224':
        try:
            eva_model = create_model('eva02_small_patch14_224',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            eva_model = create_model('eva02_small_patch14_224',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif eva_type == 'eva02_base_patch14_224':
        try:
            eva_model = create_model('eva02_base_patch14_224',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            eva_model = create_model('eva02_base_patch14_224',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif eva_type == 'eva02_large_patch14_224':
        try:
            eva_model = create_model('eva02_large_patch14_224',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            eva_model = create_model('eva02_large_patch14_224',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif eva_type == 'eva02_tiny_patch14_336':
        try:
            eva_model = create_model('eva02_tiny_patch14_336',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            eva_model = create_model('eva02_tiny_patch14_336',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif eva_type == 'eva02_small_patch14_336':
        try:
            eva_model = create_model('eva02_small_patch14_336',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            eva_model = create_model('eva02_small_patch14_336',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif eva_type == 'eva02_base_patch14_448':
        try:
            eva_model = create_model('eva02_base_patch14_448',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            eva_model = create_model('eva02_base_patch14_448',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif eva_type == 'eva02_large_patch14_448':
        try:
            eva_model = create_model('eva02_large_patch14_448',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            eva_model = create_model('eva02_large_patch14_448',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif eva_type == 'eva_giant_patch14_clip_224':
        try:
            eva_model = create_model('eva_giant_patch14_clip_224',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            eva_model = create_model('eva_giant_patch14_clip_224',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif eva_type == 'eva02_base_patch16_clip_224':
        try:
            eva_model = create_model('eva02_base_patch16_clip_224',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            eva_model = create_model('eva02_base_patch16_clip_224',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif eva_type == 'eva02_large_patch14_clip_224':
        try:
            eva_model = create_model('eva02_large_patch14_clip_224',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            eva_model = create_model('eva02_large_patch14_clip_224',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif eva_type == 'eva02_large_patch14_clip_336':
        try:
            eva_model = create_model('eva02_large_patch14_clip_336',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            eva_model = create_model('eva02_large_patch14_clip_336',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif eva_type == 'eva02_enormous_patch14_clip_224':
        try:
            eva_model = create_model('eva02_enormous_patch14_clip_224',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            eva_model = create_model('eva02_enormous_patch14_clip_224',
                                     pretrained=False,
                                     num_classes=num_classes)
    else:
        # Raise a ValueError if an unknown Eva architecture is specified
        raise ValueError(f'Unknown Eva Architecture: {eva_type}')

    # Return the created Eva model
    return eva_model
