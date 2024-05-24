from timm import create_model


def get_efficientvit_mit_model(efficientvit_mit_type, num_classes):
    """
    Returns an EfficientViT model based on the specified type and number of classes.

    Args:
        efficientvit_mit_type (str): Type of EfficientViT model. Supported types are
            'efficientvit_b0', 'efficientvit_b1', 'efficientvit_b2', 'efficientvit_b3',
            'efficientvit_l1', 'efficientvit_l2', 'efficientvit_l3'.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: EfficientViT model instance.

    Raises:
        ValueError: If the specified EfficientViT model type is not recognized.
    """
    supported_types = {
        "efficientvit_b0",
        "efficientvit_b1",
        "efficientvit_b2",
        "efficientvit_b3",
        "efficientvit_l1",
        "efficientvit_l2",
        "efficientvit_l3",
    }

    if efficientvit_mit_type not in supported_types:
        msg = f"Unknown EfficientViT-MIT Architecture: {efficientvit_mit_type}"
        raise ValueError(msg)

    try:
        return create_model(
            efficientvit_mit_type, pretrained=True, num_classes=num_classes,
        )
    except RuntimeError as e:
        print(f"{efficientvit_mit_type} - Error loading pretrained model: {e}")
        return create_model(
            efficientvit_mit_type, pretrained=False, num_classes=num_classes,
        )
