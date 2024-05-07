from timm import create_model


def get_edgenet_model(edgenet_type, num_classes):
    """
    Retrieves a pre-trained EdgeNet model based on the specified architecture.

    Parameters:
    - edgenet_type (str): The type of EdgeNet architecture to use. Supported options are:
        - "edgenext_xx_small"
        - "edgenext_x_small"
        - "edgenext_small"
        - "edgenext_base"
        - "edgenext_small_rw"

    - num_classes (int): The number of output classes for the classification task.

    Returns:
    - edgenet_model: A pre-trained EdgeNet model with the specified architecture and number of classes.

    Raises:
    - ValueError: If the provided edgenet_type is not one of the supported options.

    Example:
    >>> model = get_edgenet_model("edgenext_small", num_classes=10)
    """
    if edgenet_type == "edgenext_xx_small":
        try:
            edgenet_model = create_model('edgenext_xx_small',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            edgenet_model = create_model('edgenext_xx_small',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif edgenet_type == "edgenext_x_small":
        try:
            edgenet_model = create_model('edgenext_x_small',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            edgenet_model = create_model('edgenext_x_small',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif edgenet_type == "edgenext_small":
        try:
            edgenet_model = create_model('edgenext_small',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            edgenet_model = create_model('edgenext_small',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif edgenet_type == "edgenext_base":
        try:
            edgenet_model = create_model('edgenext_base',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            edgenet_model = create_model('edgenext_base',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif edgenet_type == "edgenext_small_rw":
        try:
            edgenet_model = create_model('edgenext_small_rw',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            edgenet_model = create_model('edgenext_small_rw',
                                         pretrained=False,
                                         num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Inception Next Architecture: {edgenet_type}')

    return edgenet_model
