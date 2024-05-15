from timm import create_model


def get_tresnet_model(tresnet_type, num_classes):
    """
    Get a TResNet model.

    Parameters:
        tresnet_type (str): Type of TResNet architecture.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: TResNet model.

    Raises:
        ValueError: If an unknown TResNet architecture is specified.
    """
    valid_types = {'tresnet_m', 'tresnet_l', 'tresnet_xl', 'tresnet_v2_l'}
    if tresnet_type not in valid_types:
        raise ValueError(f'Unknown TResNet Architecture: {tresnet_type}')

    try:
        return create_model(
            tresnet_type,
            pretrained=True,
            num_classes=num_classes)
    except RuntimeError as e:
        print(f"{tresnet_type} - Error loading pretrained model: {e}")
        return create_model(
            tresnet_type,
            pretrained=False,
            num_classes=num_classes)
