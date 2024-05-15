from timm import create_model


def get_xception_model(xception_type, num_classes):
    """
    Load a pre-trained Xception model of the specified type and modify its
    last layer to accommodate the given number of classes.

    Parameters:
    - xception_type (str): Type of Xception architecture.
    - num_classes (int): Number of output classes for the modified last layer.

    Returns:
    - torch.nn.Module: Modified Xception model with the specified architecture
      and last layer adapted for the given number of classes.

    Raises:
    - ValueError: If an unknown Xception architecture type is provided.
    """
    valid_types = {
        'legacy_xception', 'xception41', 'xception65', 'xception71',
        'xception41p', 'xception65p'
    }

    if xception_type not in valid_types:
        raise ValueError(f'Unknown Xception Architecture: {xception_type}')

    try:
        return create_model(
            xception_type,
            pretrained=True,
            num_classes=num_classes)
    except RuntimeError as e:
        print(f"{xception_type} - Error loading pretrained model: {e}")
        return create_model(
            xception_type,
            pretrained=False,
            num_classes=num_classes)
