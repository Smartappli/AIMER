from timm import create_model


def get_efficientformer_model(efficientformer_type, num_classes):
    """
    Function to get an Efficientformer model of a specified type.

    Parameters:
        efficientformer_type (str): Type of Efficientformer model to be used.
                                    Choices: 'efficientformer_l1', 'efficientformer_l3', 'efficientformer_l7'.
        num_classes (int): Number of classes for the classification task.

    Returns:
        efficientformer_model: Efficientformer model instance with specified architecture and number of classes.

    Raises:
        ValueError: If the specified efficientformer_type is not one of the supported architectures.
    """
    if efficientformer_type == 'efficientformer_l1':
        try:
            efficientformer_model = create_model('efficientformer_l1',
                                                 pretrained=True,
                                                 num_classes=num_classes)
        except:
            efficientformer_model = create_model('efficientformer_l1',
                                                 pretrained=False,
                                                 num_classes=num_classes)
    elif efficientformer_type == 'efficientformer_l3':
        try:
            efficientformer_model = create_model('efficientformer_l3',
                                                 pretrained=True,
                                                 num_classes=num_classes)
        except:
            efficientformer_model = create_model('efficientformer_l3',
                                                 pretrained=False,
                                                 num_classes=num_classes)
    elif efficientformer_type == 'efficientformer_l7':
        try:
            efficientformer_model = create_model('efficientformer_l7',
                                                 pretrained=True,
                                                 num_classes=num_classes)
        except:
            efficientformer_model = create_model('efficientformer_l7',
                                                 pretrained=False,
                                                 num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Efficientformer Architecture: {efficientformer_type}')

    return efficientformer_model