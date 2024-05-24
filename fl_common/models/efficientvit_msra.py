from timm import create_model


def get_efficientvit_msra_model(efficientvit_msra_type, num_classes):
    """
    Get an EfficientViT-MSRA model of specified type.

    Args:
        efficientvit_msra_type (str): Type of EfficientViT-MSRA model.
            It should be one of ['efficientvit_m0', 'efficientvit_m1', 'efficientvit_m2', 'efficientvit_m3',
            'efficientvit_m4', 'efficientvit_m5'].
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: EfficientViT-MSRA model.
    """
    supported_types = {
        "efficientvit_m0",
        "efficientvit_m1",
        "efficientvit_m2",
        "efficientvit_m3",
        "efficientvit_m4",
        "efficientvit_m5",
    }

    if efficientvit_msra_type not in supported_types:
        msg = f"Unknown EfficientViT-MSRA Architecture: {efficientvit_msra_type}"
        raise ValueError(msg)

    try:
        return create_model(
            efficientvit_msra_type, pretrained=True, num_classes=num_classes,
        )
    except RuntimeError as e:
        print(f"{efficientvit_msra_type} - Error loading pretrained model: {e}")
        return create_model(
            efficientvit_msra_type, pretrained=False, num_classes=num_classes,
        )
