from timm import create_model


def get_repghost_model(repghost_type, num_classes):
    """
    Returns a Residual-Path Ghost Network (RepGhost) model based on the provided RepGhost type and number of classes.

    Args:
    - repghost_type (str): The type of RepGhost model.
    - num_classes (int): The number of output classes for the model.

    Returns:
    - repghost_model: The RepGhost model instantiated based on the specified architecture.

    Raises:
    - ValueError: If the provided repghost_type is not recognized.
    """
    valid_types = [
        "repghostnet_050",
        "repghostnet_058",
        "repghostnet_080",
        "repghostnet_100",
        "repghostnet_111",
        "repghostnet_130",
        "repghostnet_150",
        "repghostnet_200",
    ]

    if repghost_type not in valid_types:
        raise ValueError(f"Unknown Repghost Architecture: {repghost_type}")

    try:
        return create_model(
            repghost_type, pretrained=True, num_classes=num_classes
        )
    except RuntimeError as e:
        print(f"{repghost_type} - Error loading pretrained model: {e}")
        return create_model(
            repghost_type, pretrained=False, num_classes=num_classes
        )
