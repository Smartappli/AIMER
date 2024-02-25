from timm import create_model


def get_pnasnet_model(pnasnet_type, num_classes):
    """
    Get a PNASNet model based on the specified architecture type.

    Args:
        pnasnet_type (str): The type of PNASNet architecture. It can be one of the following:
            - 'pnasnet5large': PNASNet-5 architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The PNASNet model.

    Raises:
        ValueError: If an unknown PNASNet architecture type is specified.
    """
    if pnasnet_type == 'pnasnet5large':
        pnasnet_model = create_model('pnasnet5large', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Pnasnet Architecture: {pnasnet_type}')

    return pnasnet_model
