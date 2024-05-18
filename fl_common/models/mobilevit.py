from timm import create_model


def get_mobilevit_model(mobilevit_type, num_classes):
    """
    Create and return an instance of the specified MobileViT architecture.

    Args:
        mobilevit_type (str): The type of MobileViT architecture to create. Options include:
                              'mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s', 'mobilevitv2_050', 'mobilevitv2_075',
                              'mobilevitv2_100', 'mobilevitv2_125', 'mobilevitv2_150', 'mobilevitv2_175',
                              'mobilevitv2_200'.
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The created instance of the specified MobileViT architecture.

    Raises:
        ValueError: If an unknown MobileViT architecture type is specified.
    """
    mobilevit_options = [
        "mobilevit_xxs",
        "mobilevit_xs",
        "mobilevit_s",
        "mobilevitv2_050",
        "mobilevitv2_075",
        "mobilevitv2_100",
        "mobilevitv2_125",
        "mobilevitv2_150",
        "mobilevitv2_175",
        "mobilevitv2_200",
    ]

    if mobilevit_type not in mobilevit_options:
        raise ValueError(f"Unknown MobileViT Architecture: {mobilevit_type}")

    try:
        return create_model(mobilevit_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{mobilevit_type} - Error loading pretrained model: {e}")
        return create_model(mobilevit_type, pretrained=False, num_classes=num_classes)
