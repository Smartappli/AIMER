from timm import create_model


def get_nest_model(nest_type, num_classes):
    """
    Get a NEST model based on the specified architecture type.

    Args:
        nest_type (str): The type of NEST architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The NEST model.

    Raises:
        ValueError: If an unknown NEST architecture type is specified.
    """
    valid_nest_types = [
        'nest_base', 'nest_small', 'nest_tiny',
        'nest_base_jx', 'nest_small_jx', 'nest_tiny_jx'
    ]

    if nest_type not in valid_nest_types:
        raise ValueError(f'Unknown NEST Architecture: {nest_type}')

    try:
        return create_model(nest_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"Error loading pretrained model: {e}")
        return create_model(nest_type, pretrained=False, num_classes=num_classes)
