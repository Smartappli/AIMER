from timm import create_model


def get_nextvit_model(nextvit_type, num_classe):
    """
    Get a NEXTVIT model based on the specified architecture type.

    Args:
        nextvit_type (str): The type of NEXTVIT architecture. It can be one of the following:
            - 'nextvit_small': Small NEXTVIT architecture.
            - 'nextvit_base': Base NEXTVIT architecture.
            - 'nextvit_large': Large NEXTVIT architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The NEXTVIT model.

    Raises:
        ValueError: If an unknown NEXTVIT architecture type is specified.
    """
    if nextvit_type == 'nextvit_small':
        nextvit_model = create_model('nextvit_small', pretrained=True, num_classes=num_classe)
    elif nextvit_type == 'nextvit_base':
        nextvit_model = create_model('nextvit_base', pretrained=True, num_classes=num_classe)
    elif nextvit_type == 'nextvit_large':
        nextvit_model = create_model('nextvit_large', pretrained=True, num_classes=num_classe)
    else:
        raise ValueError(f'Unknown Nextvit Architecture: {nextvit_type}')

    return nextvit_model