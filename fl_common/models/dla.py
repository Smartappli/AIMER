from timm import create_model


def get_dla_model(dla_type, num_classes):
    """
    Creates and returns a DLA (Deep Layer Aggregation) model based on the specified architecture type.

    Parameters:
        dla_type (str): Type of DLA architecture to use.
        num_classes (int): Number of output classes.

    Returns:
        dla_model: A DLA model instance based on the specified architecture type.

    Raises:
        ValueError: If an unknown DLA architecture type is specified.
    """
    valid_dla_types = [
        "dla60_res2net", "dla60_res2next", "dla34", "dla46_c",
        "dla46x_c", "dla60x_c", "dla60", "dla60x", "dla102",
        "dla102x", "dla102x2", "dla169"
    ]

    if dla_type not in valid_dla_types:
        raise ValueError(f'Unknown DLA Architecture: {dla_type}')

    try:
        return create_model(dla_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{dla_type} - Error loading pretrained model: {e}")
        return create_model(
            dla_type,
            pretrained=False,
            num_classes=num_classes)
