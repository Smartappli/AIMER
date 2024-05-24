from timm import create_model


def get_efficientformer_v2_model(efficientformer_v2_type, num_classes):
    """
    Get an Efficientformer v2 model of specified type.

    Args:
        efficientformer_v2_type (str): Type of Efficientformer v2 model.
            It should be one of ['efficientformerv2_s0', 'efficientformerv2_s1', 'efficientformerv2_s2', 'efficientformerv2_l'].
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Efficientformer v2 model.
    """
    supported_types = [
        "efficientformerv2_s0",
        "efficientformerv2_s1",
        "efficientformerv2_s2",
        "efficientformerv2_l",
    ]
    if efficientformer_v2_type not in supported_types:
        msg = f"Unknown Efficientformer v2 Architecture: {efficientformer_v2_type}"
        raise ValueError(msg)

    try:
        return create_model(
            efficientformer_v2_type, pretrained=True, num_classes=num_classes,
        )
    except RuntimeError as e:
        print(f"{efficientformer_v2_type} - Error loading pretrained model: {e}")
        return create_model(
            efficientformer_v2_type, pretrained=False, num_classes=num_classes,
        )
