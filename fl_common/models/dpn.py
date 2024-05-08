from timm import create_model


def get_dpn_model(dpn_type, num_classes):
    """
    Creates and returns a DPN (Dual-Path Network) model based on the specified architecture type.

    Parameters:
        dpn_type (str): Type of DPN architecture to use. Options: "dpn48b", "dpn68", "dpn68b", "dpn92", "dpn98", "dpn131", "dpn107".
        num_classes (int): Number of output classes.

    Returns:
        dpn_model: A DPN model instance based on the specified architecture type.

    Raises:
        ValueError: If an unknown DPN architecture type is specified.
    """
    if dpn_type == "dpn48b":
        try:
            dpn_model = create_model('dpn48b',
                                     pretrained=True,
                                     num_classes=num_classes)
        except OSError:
            dpn_model = create_model('dpn48b',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif dpn_type == "dpn68":
        try:
            dpn_model = create_model('dpn68',
                                     pretrained=True,
                                     num_classes=num_classes)
        except OSError:
            dpn_model = create_model('dpn68',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif dpn_type == "dpn68b":
        try:
            dpn_model = create_model('dpn68b',
                                     pretrained=True,
                                     num_classes=num_classes)
        except OSError:
            dpn_model = create_model('dpn68b',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif dpn_type == "dpn92":
        try:
            dpn_model = create_model('dpn92',
                                     pretrained=True,
                                     num_classes=num_classes)
        except OSError:
            dpn_model = create_model('dpn92',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif dpn_type == "dpn98":
        try:
            dpn_model = create_model('dpn98',
                                     pretrained=True,
                                     num_classes=num_classes)
        except OSError:
            dpn_model = create_model('dpn98',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif dpn_type == "dpn131":
        try:
            dpn_model = create_model('dpn131',
                                     pretrained=True,
                                     num_classes=num_classes)
        except OSError:
            dpn_model = create_model('dpn131',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif dpn_type == "dpn107":
        try:
            dpn_model = create_model('dpn107',
                                     pretrained=True,
                                     num_classes=num_classes)
        except OSError:
            dpn_model = create_model('dpn107',
                                     pretrained=False,
                                     num_classes=num_classes)
    else:
        # Raise a ValueError if an unknown DPN architecture is specified
        raise ValueError(f'Unknown DPN Architecture: {dpn_type}')

    # Return the created DPN model
    return dpn_model
