from timm import create_model


def get_byobnet_model(byobnet_type, num_classes):
    """
    Creates a Byobnet model according to the specified type.

    Args:
        byobnet_type (str): The type of Byobnet model to create.
        num_classes (int): The number of classes for the classification task.

    Returns:
        torch.nn.Module: The created Byobnet model with the specified number of classes.

    Raises:
        ValueError: If the specified Byobnet model type is not recognized.
    """
    model_types = [
        'gernet_l',
        'gernet_m',
        'gernet_s',
        'repvgg_a0',
        'repvgg_a1',
        'repvgg_a2',
        'repvgg_b0',
        'repvgg_b1',
        'repvgg_b1g4',
        'repvgg_b2',
        'repvgg_b2g4',
        'repvgg_b3',
        'repvgg_b3g4',
        'repvgg_d2se',
        'resnet51q',
        'resnet61q',
        'resnext26ts',
        'gcresnext26ts',
        'seresnext26ts',
        'eca_resnext26ts',
        'bat_resnext26ts',
        'resnet32ts',
        'resnet33ts',
        'gcresnet33ts',
        'seresnet33ts',
        'eca_resnet33ts',
        'gcresnet50t',
        'gcresnext50ts',
        'regnetz_b16',
        'regnetz_c16',
        'regnetz_d32',
        'regnetz_d8',
        'regnetz_e8',
        'regnetz_b16_evos',
        'regnetz_c16_evos',
        'regnetz_d8_evos',
        'mobileone_s0',
        'mobileone_s1',
        'mobileone_s2',
        'mobileone_s3',
        "mobileone_s4"]

    if byobnet_type not in model_types:
        raise ValueError(
            f"The Byobnet model type '{byobnet_type}' is not recognized.")

    try:
        return create_model(
            byobnet_type,
            pretrained=True,
            num_classes=num_classes)
    except RuntimeError as e:
        print(f"{byobnet_type} - Error loading pretrained model: {e}")
        return create_model(
            byobnet_type,
            pretrained=False,
            num_classes=num_classes)
