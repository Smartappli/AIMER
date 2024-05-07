from timm import create_model


def get_convmixer_model(convmixer_type, num_classes):
    """
    Creates and returns a Convmixer model based on the specified architecture type.

    Parameters:
        convmixer_type (str): The type of Convmixer architecture to use. It can be one of the following:
                              - "convmixer_1536_20"
                              - "convmixer_768_32"
                              - "convmixer_1024_20_ks9_p14"
        num_classes (int): The number of classes for the final classification layer.

    Returns:
        torch.nn.Module: The Convmixer model with the specified architecture and number of classes.

    Raises:
        ValueError: If the specified Convmixer architecture type is unknown.
    """
    if convmixer_type == "convmixer_1536_20":
        try:
            convmixer_model = create_model('convmixer_1536_20',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            convmixer_model = create_model('convmixer_1536_20',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif convmixer_type == "convmixer_768_32":
        try:
            convmixer_model = create_model('convmixer_768_32',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            convmixer_model = create_model('convmixer_768_32',
                                           pretrained=False,
                                           num_classes=num_classes)
    elif convmixer_type == "convmixer_1024_20_ks9_p14":
        try:
            convmixer_model = create_model('convmixer_1024_20_ks9_p14',
                                           pretrained=True,
                                           num_classes=num_classes)
        except ValueError:
            convmixer_model = create_model('convmixer_1024_20_ks9_p14',
                                           pretrained=False,
                                           num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Convmixer Architecture: {convmixer_type}')

    return convmixer_model
