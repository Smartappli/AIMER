from timm import create_model


def get_senet_model(senet_type, num_classes):
    """
    Get a SENet model based on the specified architecture type.

    Args:
        senet_type (str): The type of SENet architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The SENet model.

    Raises:
        ValueError: If an unknown SENet architecture type is specified.
    """
    valid_types = [
        'legacy_seresnet18', 'legacy_seresnet34', 'legacy_seresnet50',
        'legacy_seresnet101', 'legacy_seresnet152', 'legacy_senet154',
        'legacy_seresnext26_32x4d', 'legacy_seresnext50_32x4d', 'legacy_seresnext101_32x4d'
    ]

    if senet_type not in valid_types:
        raise ValueError(f'Unknown Senet Architecture: {senet_type}')

    try:
        return create_model(senet_type, pretrained=False, num_classes=num_classes)
    except RuntimeError as e:
        print(f"Error loading pretrained model: {e}")
        return create_model(senet_type, pretrained=False, num_classes=num_classes)
