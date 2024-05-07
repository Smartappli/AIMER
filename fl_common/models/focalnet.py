from timm import create_model


def get_focalnet_model(focalnet_type, num_classes):
    """
    Creates and returns a Focalnet model based on the specified architecture type.

    Parameters:
        focalnet_type (str): Type of Focalnet architecture to use. Options include variations like
                             "focalnet_tiny_srf", "focalnet_small_srf", "focalnet_base_srf",
                             "focalnet_tiny_lrf", "focalnet_small_lrf", "focalnet_base_lrf",
                             "focalnet_large_fl3", "focalnet_large_fl4", "focalnet_xlarge_fl3",
                             "focalnet_xlarge_fl4", "focalnet_huge_fl3", "focalnet_huge_fl4".
        num_classes (int): Number of output classes.

    Returns:
        focalnet_model: A Focalnet model instance based on the specified architecture type.

    Raises:
        ValueError: If an unknown Focalnet architecture type is specified.
    """
    if focalnet_type == 'focalnet_tiny_srf':
        try:
            focalnet_model = create_model('focalnet_tiny_srf',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            focalnet_model = create_model('focalnet_tiny_srf',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif focalnet_type == 'focalnet_small_srf':
        try:
            focalnet_model = create_model('focalnet_small_srf',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            focalnet_model = create_model('focalnet_small_srf',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif focalnet_type == 'focalnet_base_srf':
        try:
            focalnet_model = create_model('focalnet_base_srf',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            focalnet_model = create_model('focalnet_base_srf',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif focalnet_type == 'focalnet_tiny_lrf':
        try:
            focalnet_model = create_model('focalnet_tiny_lrf',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            focalnet_model = create_model('focalnet_base_srf',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif focalnet_type == 'focalnet_small_lrf':
        try:
            focalnet_model = create_model('focalnet_small_lrf',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            focalnet_model = create_model('focalnet_small_lrf',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif focalnet_type == 'focalnet_base_lrf':
        try:
            focalnet_model = create_model('focalnet_base_lrf',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            focalnet_model = create_model('focalnet_base_lrf',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif focalnet_type == 'focalnet_large_fl3':
        try:
            focalnet_model = create_model('focalnet_large_fl3',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            focalnet_model = create_model('focalnet_large_fl3',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif focalnet_type == 'focalnet_large_fl4':
        try:
            focalnet_model = create_model('focalnet_large_fl4',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            focalnet_model = create_model('focalnet_large_fl4',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif focalnet_type == 'focalnet_xlarge_fl3':
        try:
            focalnet_model = create_model('focalnet_xlarge_fl3',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            focalnet_model = create_model('focalnet_xlarge_fl3',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif focalnet_type == 'focalnet_xlarge_fl4':
        try:
            focalnet_model = create_model('focalnet_xlarge_fl4',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            focalnet_model = create_model('focalnet_xlarge_fl4',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif focalnet_type == 'focalnet_huge_fl3':
        try:
            focalnet_model = create_model('focalnet_huge_fl3',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            focalnet_model = create_model('focalnet_huge_fl3',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif focalnet_type == 'focalnet_huge_fl4':
        try:
            focalnet_model = create_model('focalnet_huge_fl4',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            focalnet_model = create_model('focalnet_huge_fl4',
                                          pretrained=False,
                                          num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Focalnet Architecture: {focalnet_type}')

    return focalnet_model
