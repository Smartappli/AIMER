from timm import create_model


def get_tnt_model(tnt_type, num_classes):
    """
    Get a TnT (Token and Token) model based on the specified architecture type.

    Args:
        tnt_type (str): The type of TnT architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The TnT model.

    Raises:
        ValueError: If an unknown TnT architecture type is specified.
    """
    valid_types = {'tnt_s_patch16_224', 'tnt_b_patch16_224'}
    if tnt_type not in valid_types:
        raise ValueError(f'Unknown TnT Architecture: {tnt_type}')

    try:
        return create_model(tnt_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"Error loading pretrained model: {e}")
        return create_model(tnt_type, pretrained=False, num_classes=num_classes)
