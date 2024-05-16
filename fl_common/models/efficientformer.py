from timm import create_model


def get_efficientformer_model(efficientformer_type, num_classes):
    """
    Function to get an Efficientformer model of a specified type.

    Parameters:
        efficientformer_type (str): Type of Efficientformer model to be used.
                                    Choices: 'efficientformer_l1', 'efficientformer_l3', 'efficientformer_l7'.
        num_classes (int): Number of classes for the classification task.

    Returns:
        efficientformer_model: Efficientformer model instance with specified architecture and number of classes.

    Raises:
        ValueError: If the specified efficientformer_type is not one of the supported architectures.
    """
    supported_types = ["efficientformer_l1", "efficientformer_l3", "efficientformer_l7"]
    if efficientformer_type not in supported_types:
        raise ValueError(
            f"Unknown Efficientformer Architecture: {efficientformer_type}"
        )

    try:
        return create_model(
            efficientformer_type, pretrained=True, num_classes=num_classes
        )
    except RuntimeError as e:
        print(f"{efficientformer_type} - Error loading pretrained model: {e}")
        return create_model(
            efficientformer_type, pretrained=False, num_classes=num_classes
        )
