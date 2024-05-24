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
    edgenet_options = [
        "edgenext_xx_small",
        "edgenext_x_small",
        "edgenext_small",
        "edgenext_base",
        "edgenext_small_rw",
    ]

    if edgenet_type not in edgenet_options:
        msg = f"Unknown EdgeNet Architecture: {edgenet_type}"
        raise ValueError(msg)

    try:
        return create_model(edgenet_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{edgenet_type} - Error loading pretrained model: {e}")
        return create_model(edgenet_type, pretrained=False, num_classes=num_classes)
