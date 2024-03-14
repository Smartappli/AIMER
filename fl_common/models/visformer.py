from timm import create_model


def get_visformer_model(visformer_type, num_classes):
    """
    Get a Visformer model based on the specified architecture type.

    Args:
        visformer_type (str): The type of Visformer architecture. It can be one of the following:
            - 'visformer_tiny': Tiny Visformer architecture.
            - 'visformer_small': Small Visformer architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The Visformer model.

    Raises:
        ValueError: If an unknown Visformer architecture type is specified.
    """
    if visformer_type == 'visformer_tiny':
        try:
            visformer_model = create_model('visformer_tiny', pretrained=True, num_classes=num_classes)
        except:
            visformer_model = create_model('visformer_tiny', pretrained=False, num_classes=num_classes)
    elif visformer_type == 'visformer_small':
        try:
            visformer_model = create_model('visformer_small', pretrained=True, num_classes=num_classes)
        except:
            visformer_model = create_model('visformer_small', pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Visformer Architecture: {visformer_type}')

    return visformer_model
