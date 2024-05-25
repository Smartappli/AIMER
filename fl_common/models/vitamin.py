from timm import create_model


def get_vitamin_model(vitamin_type, num_classes):
    """
    Retrieves a pre-trained ViTamin model based on the specified architecture.

    Parameters:
    - vitamin_type (str): The type of vitamin architecture to be retrieved. Must be one of the valid types.
    - num_classes (int): The number of output classes for the classification task. Must be a positive integer.

    Returns:
    - torch.nn.Module: A pre-trained vitamin model with the specified architecture and number of classes.

    Raises:
    - ValueError: If the specified `vitamin_type` is not one of the supported architectures.
    """
    valid_types = {
        "vitamin_small",
        "vitamin_base",
        "vitamin_large",
        "vitamin_large_256",
        "vitamin_large_336",
        "vitamin_large_384",
        "vitamin_xlarge_256",
        "vitamin_xlarge_336",
        "vitamin_xlarge_384",
    }

    if vitamin_type not in valid_types:
        msg = f"Unknown vitamin Architecture: {vitamin_type}"
        raise ValueError(msg)

    try:
        return create_model(
            vitamin_type, pretrained=True, num_classes=num_classes,
        )
    except RuntimeError as e:
        print(f"{vitamin_type} - Error loading pretrained model: {e}")
        return create_model(
            vitamin_type, pretrained=False, num_classes=num_classes,
        )
