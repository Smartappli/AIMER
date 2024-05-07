from timm import create_model


def get_dla_model(dla_type, num_classes):
    """
    Creates and returns a DLA (Deep Layer Aggregation) model based on the specified architecture type.

    Parameters:
        dla_type (str): Type of DLA architecture to use. Options include: "dla60_res2net", "dla60_res2next",
                        "dla34", "dla46_c", "dla46x_c", "dla60x_c", "dla60", "dla60x", "dla102", "dla102x",
                        "dla102x2", "dla169".
        num_classes (int): Number of output classes.

    Returns:
        dla_model: A DLA model instance based on the specified architecture type.

    Raises:
        ValueError: If an unknown DLA architecture type is specified.
    """
    if dla_type == "dla60_res2net":
        try:
            dla_model = create_model('dla60_res2net',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            dla_model = create_model('dla60_res2net',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif dla_type == "dla60_res2next":
        try:
            dla_model = create_model('dla60_res2next',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            dla_model = create_model('dla60_res2next',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif dla_type == "dla34":
        try:
            dla_model = create_model('dla34',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            dla_model = create_model('dla34',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif dla_type == "dla46_c":
        try:
            dla_model = create_model('dla46_c',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            dla_model = create_model('dla46_c',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif dla_type == "dla46x_c":
        try:
            dla_model = create_model('dla46x_c',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            dla_model = create_model('dla46x_c',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif dla_type == "dla60x_c":
        try:
            dla_model = create_model('dla60x_c',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            dla_model = create_model('dla60x_c',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif dla_type == "dla60":
        try:
            dla_model = create_model('dla60',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            dla_model = create_model('dla60',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif dla_type == "dla60x":
        try:
            dla_model = create_model('dla60x',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            dla_model = create_model('dla60x',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif dla_type == "dla102":
        try:
            dla_model = create_model('dla102',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            dla_model = create_model('dla102',
                                     pretrained=False,
                                     m_classes=num_classes)
    elif dla_type == "dla102x":
        try:
            dla_model = create_model('dla102x2',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            dla_model = create_model('dla102x2',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif dla_type == "dla102x2":
        try:
            dla_model = create_model('dla102x2',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            dla_model = create_model('dla102x2',
                                     pretrained=False,
                                     num_classes=num_classes)
    elif dla_type == "dla169":
        try:
            dla_model = create_model('dla169',
                                     pretrained=True,
                                     num_classes=num_classes)
        except ValueError:
            dla_model = create_model('dla169',
                                     pretrained=False,
                                     num_classes=num_classes)
    else:
        # Raise a ValueError if an unknown DLA architecture is specified
        raise ValueError(f'Unknown DLA Architecture: {dla_type}')

    # Return the created DLA model
    return dla_model
