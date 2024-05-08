from timm import create_model


def get_mlp_mixer_model(mlp_mixer_type, num_classes):
    """
    Create and return an instance of the specified MLP Mixer architecture.

    Args:
        mlp_mixer_type (str): The type of MLP Mixer architecture to create. Options include:
                              'mixer_s32_224', 'mixer_s16_224', 'mixer_b32_224', 'mixer_b16_224', 'mixer_l32_224',
                              'mixer_l16_224', 'gmixer_12_224', 'gmixer_24_224', 'resmlp_12_224', 'resmlp_24_224',
                              'resmlp_36_224', 'resmlp_big_24_224', 'gmlp_ti16_224', 'gmlp_s16_224', 'gmlp_b16_224'.
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The created instance of the specified MLP Mixer architecture.

    Raises:
        ValueError: If an unknown MLP Mixer architecture type is specified.
    """
    mlp_mixer_options = [
        'mixer_s32_224', 'mixer_s16_224', 'mixer_b32_224', 'mixer_b16_224', 'mixer_l32_224',
        'mixer_l16_224', 'gmixer_12_224', 'gmixer_24_224', 'resmlp_12_224', 'resmlp_24_224',
        'resmlp_36_224', 'resmlp_big_24_224', 'gmlp_ti16_224', 'gmlp_s16_224', 'gmlp_b16_224'
    ]

    if mlp_mixer_type not in mlp_mixer_options:
        raise ValueError(f'Unknown MLP Mixer Architecture: {mlp_mixer_type}')

    try:
        return create_model(mlp_mixer_type, pretrained=True, num_classes=num_classes)
    except OSError as e:
        print(f"Error loading pretrained model: {e}")
        return create_model(mlp_mixer_type, pretrained=False, num_classes=num_classes)
