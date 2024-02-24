from timm import create_model


def get_sknet_model(sknet_type, num_classes):
    """
    Get an SKNet model based on the specified architecture type.

    Args:
        sknet_type (str): The type of SKNet architecture. It can be one of the following:
            - 'skresnet18': SKResNet-18 architecture.
            - 'skresnet34': SKResNet-34 architecture.
            - 'skresnet50': SKResNet-50 architecture.
            - 'skresnet50d': SKResNet-50d architecture.
            - 'skresnext50_32x4d': SKResNeXt-50 32x4d architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The SKNet model.

    Raises:
        ValueError: If an unknown SKNet architecture type is specified.
    """
    if sknet_type == 'skresnet18':
        sknet_model = create_model('skresnet18', pretrained=True, num_classes=num_classes)
    elif sknet_type == 'skresnet34':
        sknet_model = create_model('skresnet34', pretrained=True, num_classes=num_classes)
    elif sknet_type == 'skresnet50':
        sknet_model = create_model('skresnet50', pretrained=False, num_classes=num_classes)
    elif sknet_type == 'skresnet50d':
        sknet_model = create_model('skresnet50d', pretrained=False, num_classes=num_classes)
    elif sknet_type == 'skresnext50_32x4d':
        sknet_model = create_model('skresnext50_32x4d', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Sknet Architecture: {sknet_type}')

    return sknet_model