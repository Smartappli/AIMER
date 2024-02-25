from timm import create_model


def get_rexnet_model(rexnet_type, num_classes):
    """
    Create and return a Rexnet model based on the specified architecture type.

    Parameters:
        rexnet_type (str): The type of Rexnet architecture to use. It can be one of the following:
                           - 'rexnet_100'
                           - 'rexnet_130'
                           - 'rexnet_150'
                           - 'rexnet_200'
                           - 'rexnet_300'
                           - 'rexnetr_100'
                           - 'rexnetr_130'
                           - 'rexnetr_150'
                           - 'rexnetr_200'
                           - 'rexnetr_300'
        num_classes (int): The number of classes for the final classification layer.

    Returns:
        torch.nn.Module: The Rexnet model with the specified architecture and number of classes.

    Raises:
        ValueError: If the specified Rexnet architecture type is unknown.
    """
    if rexnet_type == 'rexnet_100':
        rexnet_model = create_model('rexnet_100', pretrained=True, num_classes=num_classes)
    elif rexnet_type == 'rexnet_130':
        rexnet_model = create_model('rexnet_130', pretrained=True, num_classes=num_classes)
    elif rexnet_type == 'rexnet_150':
        rexnet_model = create_model('rexnet_150', pretrained=True, num_classes=num_classes)
    elif rexnet_type == 'rexnet_200':
        rexnet_model = create_model('rexnet_200', pretrained=True, num_classes=num_classes)
    elif rexnet_type == 'rexnet_300':
        rexnet_model = create_model('rexnet_300', pretrained=False, num_classes=num_classes)
    elif rexnet_type == 'rexnetr_100':
        rexnet_model = create_model('rexnetr_100', pretrained=False, num_classes=num_classes)
    elif rexnet_type == 'rexnetr_130':
        rexnet_model = create_model('rexnetr_130', pretrained=False, num_classes=num_classes)
    elif rexnet_type == 'rexnetr_150':
        rexnet_model = create_model('rexnetr_150', pretrained=False, num_classes=num_classes)
    elif rexnet_type == 'rexnetr_200':
        rexnet_model = create_model('rexnetr_200', pretrained=True, num_classes=num_classes)
    elif rexnet_type == 'rexnetr_300':
        rexnet_model = create_model('rexnetr_300', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Rexnet Architecture: {rexnet_type}')

    return rexnet_model