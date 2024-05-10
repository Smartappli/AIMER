from timm import create_model


def get_nasnet_model(nasnet_type, num_classes):
    """
    Get a NASNet model.

    Parameters:
        nasnet_type (str): Type of the NASNet model. Currently supports 'nasnetalarge'.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: The selected NASNet model.

    Raises:
        ValueError: If an unknown NASNet architecture is specified.
    """
    valid_nasnet_types = ['nasnetalarge']

    if nasnet_type not in valid_nasnet_types:
        raise ValueError(f'Unknown Nasnet Architecture: {nasnet_type}')

    try:
        return create_model(nasnet_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{nasnet_type} - Error loading pretrained model: {e}")
        return create_model(nasnet_type, pretrained=False, num_classes=num_classes)
