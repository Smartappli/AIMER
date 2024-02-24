from timm import create_model


def get_mobilevit_model(mobilevit_type, num_classes):
    """
    Create and return an instance of the specified MobileViT architecture.

    Args:
    - mobilevit_type (str): The type of MobileViT architecture to create. It should be one of the following:
                            'mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s', 'mobilevitv2_050', 'mobilevitv2_075',
                            'mobilevitv2_100', 'mobilevitv2_125', 'mobilevitv2_150', 'mobilevitv2_175',
                            'mobilevitv2_200'.
    - num_classes (int): The number of output classes for the model.

    Returns:
    - torch.nn.Module: The created instance of the specified MobileViT architecture.

    Raises:
    - ValueError: If an unknown MobileViT architecture type is specified.
    """
    if mobilevit_type == 'mobilevit_xxs':
        mobilevit_model = create_model('mobilevit_xxs', pretrained=True, num_classes=num_classes)
    elif mobilevit_type == 'mobilevit_xs':
        mobilevit_model = create_model('mobilevit_xs', pretrained=True, num_classes=num_classes)
    elif mobilevit_type == 'mobilevit_s':
        mobilevit_model = create_model('mobilevit_s', pretrained=True, num_classes=num_classes)
    elif mobilevit_type == 'mobilevitv2_050':
        mobilevit_model = create_model('mobilevitv2_050', pretrained=True, num_classes=num_classes)
    elif mobilevit_type == 'mobilevitv2_075':
        mobilevit_model = create_model('mobilevitv2_075', pretrained=True, num_classes=num_classes)
    elif mobilevit_type == 'mobilevitv2_100':
        mobilevit_model = create_model('mobilevitv2_100', pretrained=True, num_classes=num_classes)
    elif mobilevit_type == 'mobilevitv2_125':
        mobilevit_model = create_model('mobilevitv2_125', pretrained=True, num_classes=num_classes)
    elif mobilevit_type == 'mobilevitv2_150':
        mobilevit_model = create_model('mobilevitv2_150', pretrained=True, num_classes=num_classes)
    elif mobilevit_type == 'mobilevitv2_175':
        mobilevit_model = create_model('mobilevitv2_175', pretrained=True, num_classes=num_classes)
    elif mobilevit_type == 'mobilevitv2_200':
        mobilevit_model = create_model('mobilevitv2_200', pretrained=True, num_classes=num_classes)
    else:
        # Raise a ValueError if an unknown Mlp Mixer architecture is specified
        raise ValueError(f'Unknown Mobilevit Architecture: {mobilevit_type}')

    # Return the created Mobilevit model
    return mobilevit_model
