from timm import create_model


def get_pvt_v2_model(pvt_v2_type, num_classes):
    """
    Get a PVTv2 model based on the specified architecture type.

    Args:
        pvt_v2_type (str): The type of PVTv2 architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The PVTv2 model.

    Raises:
        ValueError: If an unknown PVTv2 architecture type is specified.
    """
    valid_pvt_v2_types = {
        'pvt_v2_b0', 'pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b3',
        'pvt_v2_b4', 'pvt_v2_b5', 'pvt_v2_b2_li'
    }

    if pvt_v2_type not in valid_pvt_v2_types:
        raise ValueError(f'Unknown PVTv2 Architecture: {pvt_v2_type}')

    try:
        return create_model(pvt_v2_type, pretrained=True, num_classes=num_classes)
    except OSError as e:
        print(f"Error loading pretrained model: {e}")
        return create_model(pvt_v2_type, pretrained=False, num_classes=num_classes)
