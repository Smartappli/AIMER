from timm import create_model


def get_dila_model(dila_type, num_classes):
    """
    Get a DILA (Differentiable Interchangeable Layer Aggregation) model.

    Parameters:
        dila_type (str): Type of DILA architecture. Options include:
            - "dla60_res2net"
            - "dla60_res2next"
            - "dla34"
            - "dla46_c"
            - "dla46x_c"
            - "dla60x_c"
            - "dla60"
            - "dla60x"
            - "dla102"
            - "dla102x"
            - "dla102x2"
            - "dla169"
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: DILA model.

    Raises:
        ValueError: If an unknown Dila architecture is specified.
    """
    if dila_type == "dla60_res2net":
        dila_model = create_model('dla60_res2net', pretrained=True, num_classes=num_classes)
    elif dila_type == "dla60_res2next":
        dila_model = create_model('dla60_res2next', pretrained=True, num_classes=num_classes)
    elif dila_type == "dla34":
        dila_model = create_model('dla34', pretrained=True, num_classes=num_classes)
    elif dila_type == "dla46_c":
        dila_model = create_model('dla46_c', pretrained=True, num_classes=num_classes)
    elif dila_type == "dla46x_c":
        dila_model = create_model('dla46x_c', pretrained=True, num_classes=num_classes)
    elif dila_type == "dla60x_c":
        dila_model = create_model('dla60x_c', pretrained=True, num_classes=num_classes)
    elif dila_type == "dla60":
        dila_model = create_model('dla60', pretrained=True, num_classes=num_classes)
    elif dila_type == "dla60x":
        dila_model = create_model('dla60x', pretrained=True, num_classes=num_classes)
    elif dila_type == "dla102":
        dila_model = create_model('dla102', pretrained=True, num_classes=num_classes)
    elif dila_type == "dla102x":
        dila_model = create_model('dla102x2', pretrained=True, num_classes=num_classes)
    elif dila_type == "dla102x2":
        dila_model = create_model('dla102x2', pretrained=True, num_classes=num_classes)
    elif dila_type == "dla169":
        dila_model = create_model('dla169', pretrained=True, num_classes=num_classes)
    else:
        # Raise a ValueError if an unknown Dila architecture is specified
        raise ValueError(f'Unknown Dila Architecture: {dila_type}')

    # Return the created Davit model
    return dila_model
