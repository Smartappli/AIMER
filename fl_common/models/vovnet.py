from timm import create_model


def get_vovnet_model(vovnet_type, num_classes):
    """
    Create and return a Vovnet model based on the specified architecture type and number of classes.

    Parameters:
    - vovnet_type (str): Type of Vovnet architecture.
    - num_classes (int): Number of output classes for the model.

    Returns:
    - vovnet_model: A pre-trained Vovnet model with the specified architecture and number of classes.

    Raises:
    - ValueError: If the provided `vovnet_type` is not recognized.
    """
    valid_types = {
        'vovnet39a', 'vovnet57a', 'ese_vovnet19b_slim_dw', 'ese_vovnet19b_dw',
        'ese_vovnet19b_slim', 'ese_vovnet39b', 'ese_vovnet57b', 'ese_vovnet99b',
        'eca_vovnet39b', 'ese_vovnet39b_evos'
    }

    if vovnet_type not in valid_types:
        raise ValueError(f'Unknown Vovnet Architecture: {vovnet_type}')

    try:
        return create_model(vovnet_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"Error loading pretrained model: {e}")
        return create_model(vovnet_type, pretrained=False, num_classes=num_classes)

