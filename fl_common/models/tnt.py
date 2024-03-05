from timm import create_model


def get_tnt_model(tnt_type, num_classes):
    """
    Get a TnT (Token and Token) model based on the specified architecture type.

    Args:
        tnt_type (str): The type of TnT architecture. It can be one of the following:
            - 'tnt_s_patch16_224': Small TnT architecture with patch size 16x16 and input size 224x224.
            - 'tnt_b_patch16_224': Big TnT architecture with patch size 16x16 and input size 224x224.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The TnT model.

    Raises:
        ValueError: If an unknown TnT architecture type is specified.
    """
    if tnt_type == 'tnt_s_patch16_224':
        try:
            tnt_model = create_model('tnt_s_patch16_224', pretrained=True, num_classes=num_classes)
        except:
            tnt_model = create_model('tnt_s_patch16_224', pretrained=False, num_classes=num_classes)
    elif tnt_type == 'tnt_b_patch16_224':
        try:
            tnt_model = create_model('tnt_b_patch16_224', pretrained=True, num_classes=num_classes)
        except:
            tnt_model = create_model('tnt_b_patch16_224', pretrained=False, num_classes=num_classes)
    else:
        # Raise a ValueError if an unknown Tnt architecture is specified
        raise ValueError(f'Unknown TinyViT Architecture: {tnt_type}')

    # Return the created Tnt model
    return tnt_model
