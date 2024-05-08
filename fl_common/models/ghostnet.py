from timm import create_model


def get_ghostnet_model(ghostnet_type, num_classes):
    """
    Get a GhostNet model based on the specified architecture type.

    Parameters:
        ghostnet_type (str): Type of GhostNet architecture.
        num_classes (int): Number of output classes.

    Returns:
        ghostnet_model: The selected GhostNet model instance.

    Raises:
        ValueError: If an unknown GhostNet architecture type is specified.
    """
    if ghostnet_type == 'ghostnet_050':
        try:
            ghostnet_model = create_model('ghostnet_050',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            ghostnet_model = create_model('ghostnet_050',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif ghostnet_type == 'ghostnet_100':
        try:
            ghostnet_model = create_model('ghostnet_100',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            ghostnet_model = create_model('ghostnet_100',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif ghostnet_type == 'ghostnet_130':
        try:
            ghostnet_model = create_model('ghostnet_130',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            ghostnet_model = create_model('ghostnet_130',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif ghostnet_type == 'ghostnetv2_100':
        try:
            ghostnet_model = create_model('ghostnetv2_100',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            ghostnet_model = create_model('ghostnetv2_100',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif ghostnet_type == 'ghostnetv2_130':
        try:
            ghostnet_model = create_model('ghostnetv2_130',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            ghostnet_model = create_model('ghostnetv2_130',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif ghostnet_type == 'ghostnetv2_160':
        try:
            ghostnet_model = create_model('ghostnetv2_160',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            ghostnet_model = create_model('ghostnetv2_130',
                                          pretrained=False,
                                          num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Ghostnet Architecture: {ghostnet_type}')

    return ghostnet_model
