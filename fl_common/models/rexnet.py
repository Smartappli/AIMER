from timm import create_model


def get_rexnet_model(rexnet_type, num_classes):
    """
    Create and return a Rexnet model based on the specified architecture type.

    Parameters:
        rexnet_type (str): The type of Rexnet architecture to use.
        num_classes (int): The number of classes for the final classification layer.

    Returns:
        torch.nn.Module: The Rexnet model with the specified architecture and number of classes.

    Raises:
        ValueError: If the specified Rexnet architecture type is unknown.
    """
    valid_types = {
        'rexnet_100', 'rexnet_130', 'rexnet_150', 'rexnet_200', 'rexnet_300',
        'rexnetr_100', 'rexnetr_130', 'rexnetr_150', 'rexnetr_200', 'rexnetr_300'
    }

    if rexnet_type not in valid_types:
        raise ValueError(f'Unknown Rexnet Architecture: {rexnet_type}')

    try:
        return create_model(rexnet_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"Error loading pretrained model: {e}")
        return create_model(rexnet_type, pretrained=False, num_classes=num_classes)
