from timm import create_model


def get_repvit_model(repvit_type, num_classes):
    """
    Returns a Residual-Path Vision Transformer (RepVIT) model based on the provided RepVIT type and number of classes.

    Args:
    - repvit_type (str): The type of RepVIT model.
    - num_classes (int): The number of output classes for the model.

    Returns:
    - repvit_model: The RepVIT model instantiated based on the specified architecture.

    Raises:
    - ValueError: If the provided repvit_type is not recognized.
    """
    valid_types = {
        'repvit_m1', 'repvit_m2', 'repvit_m3', 'repvit_m0_9',
        'repvit_m1_0', 'repvit_m1_1', 'repvit_m1_5', 'repvit_m2_3'
    }

    if repvit_type not in valid_types:
        raise ValueError(f'Unknown RepVIT Architecture: {repvit_type}')

    try:
        return create_model(repvit_type, pretrained=True, num_classes=num_classes)
    except OSError as e:
        print(f"Error loading pretrained model: {e}")
        return create_model(repvit_type, pretrained=False, num_classes=num_classes)
