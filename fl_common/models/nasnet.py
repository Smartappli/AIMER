from timm import create_model


def get_nasnet_model(nasnet_type, num_classes):
    """
    Get a NASNet model.

    Parameters:
        nasnet_type (str): Type of the NASNet model.
            Options:
                - 'nasnetalarge'
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: The selected NASNet model.

    Raises:
        ValueError: If an unknown NASNet architecture is specified.
    """
    if nasnet_type == 'nasnetalarge':
        try:
            nasnet_model = create_model('nasnetalarge',
                                        pretrained=True,
                                        num_classes=num_classes)
        except OSError:
            nasnet_model = create_model('nasnetalarge',
                                        pretrained=False,
                                        num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Nasnet Architecture: {nasnet_type}')

    return nasnet_model
