from timm import create_model


def get_res2net_model(res2net_type, num_classes):
    """
    Returns a Res2Net model based on the provided Res2Net type and number of classes.

    Args:
    - res2net_type (str): The type of Res2Net model.
    - num_classes (int): The number of output classes for the model.

    Returns:
    - res2net_model: The Res2Net model instantiated based on the specified architecture.

    Raises:
    - ValueError: If the provided res2net_type is not recognized.
    """
    valid_types = {
        'res2net50_26w_4s', 'res2net101_26w_4s', 'res2net50_26w_6s',
        'res2net50_26w_8s', 'res2net50_48w_2s', 'res2net50_14w_8s',
        'res2next50', 'res2net50d', 'res2net101d'
    }

    if res2net_type not in valid_types:
        raise ValueError(f'Unknown Res2Net Architecture: {res2net_type}')

    try:
        return create_model(res2net_type, pretrained=True, num_classes=num_classes)
    except OSError as e:
        print(f"Error loading pretrained model: {e}")
        return create_model(res2net_type, pretrained=False, num_classes=num_classes)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
