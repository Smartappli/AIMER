from timm import create_model


def get_repvit_model(repvit_type, num_classes):
    """
    Returns a Residual-Path Vision Transformer (RepVIT) model based on the provided RepVIT type and number of classes.

    Args:
    - repvit_type (str): The type of RepVIT model. It should be one of the following:
        - 'repvit_m1'
        - 'repvit_m2'
        - 'repvit_m3'
        - 'repvit_m0_9'
        - 'repvit_m1_0'
        - 'repvit_m1_1'
        - 'repvit_m1_5'
        - 'repvit_m2_3'
    - num_classes (int): The number of output classes for the model.

    Returns:
    - repvit_model: The RepVIT model instantiated based on the specified architecture.

    Raises:
    - ValueError: If the provided repvit_type is not recognized.
    """
    if repvit_type == 'repvit_m1':
        try:
            repvit_model = create_model('repvit_m1',
                                        pretrained=True,
                                        num_classes=num_classes)
        except Exception:
            repvit_model = create_model('repvit_m1',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif repvit_type == 'repvit_m2':
        try:
            repvit_model = create_model('repvit_m2',
                                        pretrained=True,
                                        num_classes=num_classes)
        except Exception:
            repvit_model = create_model('repvit_m2',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif repvit_type == 'repvit_m3':
        try:
            repvit_model = create_model('repvit_m3',
                                        pretrained=True,
                                        num_classes=num_classes)
        except Exception:
            repvit_model = create_model('repvit_m3',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif repvit_type == 'repvit_m0_9':
        try:
            repvit_model = create_model('repvit_m0_9',
                                        pretrained=True,
                                        num_classes=num_classes)
        except Exception:
            repvit_model = create_model('repvit_m3',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif repvit_type == 'repvit_m1_0':
        try:
            repvit_model = create_model('repvit_m1_0',
                                        pretrained=True,
                                        num_classes=num_classes)
        except Exception:
            repvit_model = create_model('repvit_m1_0',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif repvit_type == 'repvit_m1_1':
        try:
            repvit_model = create_model('repvit_m1_1',
                                        pretrained=True,
                                        num_classes=num_classes)
        except Exception:
            repvit_model = create_model('repvit_m1_1',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif repvit_type == 'repvit_m1_5':
        try:
            repvit_model = create_model('repvit_m1_5',
                                        pretrained=True,
                                        num_classes=num_classes)
        except Exception:
            repvit_model = create_model('repvit_m1_5',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif repvit_type == 'repvit_m2_3':
        try:
            repvit_model = create_model('repvit_m2_3',
                                        pretrained=True,
                                        num_classes=num_classes)
        except Exception:
            repvit_model = create_model('repvit_m2_3',
                                        pretrained=False,
                                        num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Repvit Architecture: {repvit_type}')

    return repvit_model
