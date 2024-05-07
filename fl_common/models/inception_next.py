from timm import create_model


def get_inception_next_model(inception_next_type, num_classes):
    """
    Get an Inception Next model based on the specified architecture type.

    Parameters:
    - inception_next_type (str): Type of Inception Next architecture. Options:
        - "inception_next_tiny"
        - "inception_next_small"
        - "inception_next_base"
    - num_classes (int): Number of output classes for the model.

    Returns:
    - torch.nn.Module: Inception Next model with the specified architecture type and number of classes.

    Raises:
    - ValueError: If an unknown Inception Next architecture type is provided.
    """
    if inception_next_type == "inception_next_tiny":
        try:
            inception_next_model = create_model('inception_next_tiny',
                                                pretrained=True,
                                                num_classes=num_classes)
        except ValueError:
            inception_next_model = create_model('inception_next_tiny',
                                                pretrained=False,
                                                num_classes=num_classes)
    elif inception_next_type == "inception_next_small":
        try:
            inception_next_model = create_model('inception_next_small',
                                                pretrained=True,
                                                num_classes=num_classes)
        except ValueError:
            inception_next_model = create_model('inception_next_small',
                                                pretrained=False,
                                                num_classes=num_classes)
    elif inception_next_type == "inception_next_base":
        try:
            inception_next_model = create_model('inception_next_base',
                                                pretrained=True,
                                                num_classes=num_classes)
        except ValueError:
            inception_next_model = create_model('inception_next_base',
                                                pretrained=False,
                                                num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Inception Next Architecture: {inception_next_type}')

    return inception_next_model
