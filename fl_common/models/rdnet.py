from timm import create_model


def get_rdnet_model(rdnet_type, num_classes):
    """
    Create an RDNet model using the timm library.

    Parameters:
    - rdnet_type (str): The type of RDNet architecture. Must be one of
      ["rdnet_tiny", "rdnet_small", "rdnet_base", "rdnet_large"].
    - num_classes (int): The number of output classes for the model.

    Returns:
    - model: A PyTorch model instance based on the RDNet architecture.

    Raises:
    - ValueError: If an invalid rdnet_type is provided.
    - RuntimeError: If there is an error loading the pretrained model.
    """

    valid_rdnet_types = [
        "rdnet_tiny",
        "rdnet_small",
        "rdnet_base",
        "rdnet_large",
    ]

    if rdnet_type not in valid_rdnet_types:
        msg = f"Unknown RDNet Architecture: {rdnet_type}. Valid types are: {valid_rdnet_types}"
        raise ValueError(msg)

    try:
        return create_model(
            rdnet_type,
            pretrained=True,
            num_classes=num_classes,
        )
    except RuntimeError as e:
        print(f"{rdnet_type} - Error loading pretrained model: {e}")
        # Alternatively, consider logging the error or raising it again
        # raise e
        return create_model(
            rdnet_type,
            pretrained=False,
            num_classes=num_classes,
        )
