from timm import create_model


def get_nest_model(nest_type, num_classes):
    """
    Get a NEST model based on the specified architecture type.

    Args:
        nest_type (str): The type of NEST architecture. It can be one of the following:
            - 'nest_base': Base NEST architecture.
            - 'nest_small': Small NEST architecture.
            - 'nest_tiny': Tiny NEST architecture.
            - 'nest_base_jx': Base NEST architecture with JX configuration.
            - 'nest_small_jx': Small NEST architecture with JX configuration.
            - 'nest_tiny_jx': Tiny NEST architecture with JX configuration.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The NEST model.

    Raises:
        ValueError: If an unknown NEST architecture type is specified.
    """
    if nest_type == 'nest_base':
        nest_model = create_model('nest_base',pretrained=True, num_classes=num_classes)
    elif nest_type == 'nest_small':
        nest_model = create_model('nest_small',pretrained=True, num_classes=num_classes)
    elif nest_type == 'nest_tiny':
        nest_model = create_model('nest_tiny',pretrained=True, num_classes=num_classes)
    elif nest_type == 'nest_base_jx':
        nest_model = create_model('nest_base_jx',pretrained=True, num_classes=num_classes)
    elif nest_type == 'nest_small_jx':
        nest_model = create_model('nest_small_jx',pretrained=True, num_classes=num_classes)
    elif nest_type == 'nest_tiny_jx':
        nest_model = create_model('nest_tiny_jx',pretrained=True,num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Nest Architecture: {nest_type}')

    return nest_model
