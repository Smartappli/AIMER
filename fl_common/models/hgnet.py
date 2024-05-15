from timm import create_model


def get_hgnet_model(hgnet_type, num_classes):
    """
    Get an HGNet (HourglassNet) model.

    Parameters:
        hgnet_type (str): Type of HGNet architecture. Options include 'hgnet_tiny', 'hgnet_small', 'hgnet_base',
                          'hgnetv2_b0' to 'hgnetv2_b6'.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: HGNet model.

    Raises:
        ValueError: If an unknown HGNet architecture is specified.
    """
    hgnet_options = ['hgnet_tiny', 'hgnet_small', 'hgnet_base'] + \
                    [f'hgnetv2_b{i}' for i in range(7)]

    if hgnet_type not in hgnet_options:
        raise ValueError(f'Unknown HGNet Architecture: {hgnet_type}')

    try:
        return create_model(
            hgnet_type,
            pretrained=True,
            num_classes=num_classes)
    except RuntimeError as e:
        print(f"{hgnet_type} - Error loading pretrained model: {e}")
        return create_model(
            hgnet_type,
            pretrained=False,
            num_classes=num_classes)
