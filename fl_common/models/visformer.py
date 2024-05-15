from timm import create_model


def get_visformer_model(visformer_type, num_classes):
    """
    Get a Visformer model based on the specified architecture type.

    Args:
        visformer_type (str): The type of Visformer architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The Visformer model.

    Raises:
        ValueError: If an unknown Visformer architecture type is specified.
    """
    valid_types = {'visformer_tiny', 'visformer_small'}
    if visformer_type not in valid_types:
        raise ValueError(f'Unknown Visformer Architecture: {visformer_type}')

    try:
        return create_model(visformer_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{visformer_type} - Error loading pretrained model: {e}")
        return create_model(visformer_type, pretrained=False, num_classes=num_classes)
