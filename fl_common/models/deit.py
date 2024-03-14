from timm import create_model


def get_deit_model(deit_type, num_classes):
    """
    Get a DeiT (Data-efficient image Transformer) model.

    Parameters:
        deit_type (str): Type of DeiT architecture. Options include:
            - "deit_tiny_patch16_224"
            - "deit_small_patch16_224"
            - "deit_base_patch16_224"
            - "deit_base_patch16_384"
            - "deit_tiny_distilled_patch16_224"
            - "deit_small_distilled_patch16_224"
            - "deit_base_distilled_patch16_224"
            - "deit_base_distilled_patch16_384"
            - "deit3_small_patch16_224"
            - "deit3_small_patch16_384"
            - "deit3_medium_patch16_224"
            - "deit3_base_patch16_224"
            - "deit3_base_patch16_384"
            - "deit3_large_patch16_224"
            - "deit3_large_patch16_384"
            - "deit3_huge_patch14_224"
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: DeiT model.

    Raises:
        ValueError: If an unknown Deit architecture is specified.
    """
    if deit_type == "deit_tiny_patch16_224":
        try:
            deit_model = create_model('deit_tiny_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            deit_model = create_model('deit_tiny_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif deit_type == "deit_small_patch16_224":
        try:
            deit_model = create_model('deit_small_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            deit_model = create_model('deit_small_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif deit_type == "deit_base_patch16_224":
        try:
            deit_model = create_model('deit_base_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            deit_model = create_model('deit_base_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif deit_type == "deit_base_patch16_384":
        try:
            deit_model = create_model('deit_base_patch16_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            deit_model = create_model('deit_base_patch16_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif deit_type == "deit_tiny_distilled_patch16_224":
        try:
            deit_model = create_model('deit_tiny_distilled_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            deit_model = create_model('deit_tiny_distilled_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif deit_type == "deit_small_distilled_patch16_224":
        try:
            deit_model = create_model('deit_small_distilled_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            deit_model = create_model('deit_small_distilled_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif deit_type == "deit_base_distilled_patch16_224":
        try:
            deit_model = create_model('deit_base_distilled_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            deit_model = create_model('deit_base_distilled_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif deit_type == "deit_base_distilled_patch16_384":
        try:
            deit_model = create_model('deit_base_distilled_patch16_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            deit_model = create_model('deit_base_distilled_patch16_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif deit_type == "deit3_small_patch16_224":
        try:
            deit_model = create_model('deit3_small_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            deit_model = create_model('deit3_small_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif deit_type == "deit3_small_patch16_384":
        try:
            deit_model = create_model('deit3_small_patch16_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            deit_model = create_model('deit3_small_patch16_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif deit_type == "deit3_medium_patch16_224":
        try:
            deit_model = create_model('deit3_medium_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            deit_model = create_model('deit3_medium_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif deit_type == "deit3_base_patch16_224":
        try:
            deit_model = create_model('deit3_base_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            deit_model = create_model('deit3_base_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif deit_type == "deit3_base_patch16_384":
        try:
            deit_model = create_model('deit3_base_patch16_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            deit_model = create_model('deit3_base_patch16_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif deit_type == "deit3_large_patch16_224":
        try:
            deit_model = create_model('deit3_large_patch16_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            deit_model = create_model('deit3_large_patch16_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif deit_type == "deit3_large_patch16_384":
        try:
            deit_model = create_model('deit3_large_patch16_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            deit_model = create_model('deit3_large_patch16_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif deit_type == "deit3_huge_patch14_224":
        try:
            deit_model = create_model('deit3_huge_patch14_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            deit_model = create_model('deit3_huge_patch14_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    else:
        # Raise a ValueError if an unknown Deit architecture is specified
        raise ValueError

    return deit_model
