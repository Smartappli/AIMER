from timm import create_model


def get_inception_next_model(inception_next_type, num_classes):
    """
    Get an Inception Next model based on the specified architecture type.

    Parameters:
        inception_next_type (str): Type of Inception Next architecture. Options:
            - "inception_next_tiny"
            - "inception_next_small"
            - "inception_next_base"
        num_classes (int): Number of output classes for the model.

    Returns:
        torch.nn.Module: Inception Next model with the specified architecture type and number of classes.

    Raises:
        ValueError: If an unknown Inception Next architecture type is provided.
    """
    inception_next_options = ["inception_next_tiny", "inception_next_small", "inception_next_base"]

    if inception_next_type not in inception_next_options:
        raise ValueError(f'Unknown Inception Next Architecture: {inception_next_type}')

    try:
        return create_model(inception_next_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{inception_next_type} - Error loading pretrained model: {e}")
        return create_model(inception_next_type, pretrained=False, num_classes=num_classes)
