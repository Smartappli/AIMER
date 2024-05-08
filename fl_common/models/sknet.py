from timm import create_model


def get_sknet_model(sknet_type, num_classes):
    """
    Get an SKNet model based on the specified architecture type.

    Args:
        sknet_type (str): The type of SKNet architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The SKNet model.

    Raises:
        ValueError: If an unknown SKNet architecture type is specified.
    """
    valid_types = {
        'skresnet18', 'skresnet34', 'skresnet50', 'skresnet50d', 'skresnext50_32x4d'
    }

    if sknet_type not in valid_types:
        raise ValueError(f'Unknown SKNet Architecture: {sknet_type}')

    try:
        return create_model(sknet_type, pretrained=True, num_classes=num_classes)
    except OSError as e:
        print(f"Error loading pretrained model: {e}")
        return create_model(sknet_type, pretrained=False, num_classes=num_classes)
