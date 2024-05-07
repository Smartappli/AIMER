from timm import create_model


def get_beit_model(beit_type, num_classes):
    """
    Retrieves a pre-trained BEiT (Bottleneck Transformers) model based on the specified architecture.

    Parameters:
    - beit_type (str): The type of BEiT architecture to be retrieved. Supported options include:
        - "beit_base_patch16_224"
        - "beit_base_patch16_384"
        - "beit_large_patch16_224"
        - "beit_large_patch16_384"
        - "beit_large_patch16_512"
        - "beitv2_base_patch16_224"
        - "beitv2_large_patch16_224"
    - num_classes (int): The number of output classes for the classification task.

    Returns:
    - torch.nn.Module: A pre-trained BEiT model with the specified architecture and number of classes.

    Raises:
    - ValueError: If the specified `beit_type` is not one of the supported architectures.
    """
    if beit_type == "beit_base_patch16_224":
        try:
            beit_model = create_model('beit_base_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            beit_model = create_model('beit_base_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif beit_type == "beit_base_patch16_384":
        try:
            beit_model = create_model('beit_base_patch16_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            beit_model = create_model('beit_base_patch16_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif beit_type == "beit_large_patch16_224":
        try:
            beit_model = create_model('beit_large_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            beit_model = create_model('beit_large_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif beit_type == "beit_large_patch16_384":
        try:
            beit_model = create_model('beit_large_patch16_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            beit_model = create_model('beit_large_patch16_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif beit_type == "beit_large_patch16_512":
        try:
            beit_model = create_model('beit_large_patch16_512',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            beit_model = create_model('beit_large_patch16_512',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif beit_type == "beitv2_base_patch16_224":
        try:
            beit_model = create_model('beitv2_base_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            beit_model = create_model('beitv2_base_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif beit_type == "beitv2_large_patch16_224":
        try:
            beit_model = create_model('beitv2_large_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            beit_model = create_model('beitv2_large_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Beit Architecture: {beit_type}')

    return beit_model
