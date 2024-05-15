from timm import create_model


def get_resnest_model(resnest_type, num_classes):
    """
    Create a ResNeSt model of specified type and number of classes.

    Args:
        resnest_type (str): Type of ResNeSt model to create.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: ResNeSt model with specified type and number of classes.

    Raises:
        ValueError: If an unknown Resnest architecture is provided.
    """
    valid_types = [
        'resnest14d', 'resnest26d', 'resnest50d', 'resnest101e',
        'resnest200e', 'resnest269e', 'resnest50d_4s2x40d', 'resnest50d_1s4x24d'
    ]

    if resnest_type not in valid_types:
        raise ValueError(f'Unknown Resnest Architecture: {resnest_type}')

    try:
        return create_model(resnest_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{resnest_type} - Error loading pretrained model: {e}")
        return create_model(resnest_type, pretrained=False, num_classes=num_classes)
