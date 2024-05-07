from timm import create_model


def get_vision_transformer_relpos_model(vision_transformer_relpos_type, num_classes):
    """
    Function to get a Vision Transformer Relative Position model of a specified type.

    Parameters:
        vision_transformer_relpos_type (str): Type of Vision Transformer Relative Position model to be used.
                                              Choices include various types such as 'vit_relpos_base_patch32_plus_rpn_256',
                                              'vit_relpos_base_patch16_plus_240', 'vit_relpos_small_patch16_224', etc.
        num_classes (int): Number of classes for the classification task.

    Returns:
        vision_transformer_relpos_model: Vision Transformer Relative Position model instance with specified architecture and number of classes.

    Raises:
        ValueError: If the specified vision_transformer_relpos_type is not one of the supported architectures.
    """
    if vision_transformer_relpos_type == "vit_relpos_base_patch32_plus_rpn_256":
        try:
            vision_transformer_relpos_model = create_model('vit_relpos_base_patch32_plus_rpn_256',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except Exception:
            vision_transformer_relpos_model = create_model('vit_relpos_base_patch32_plus_rpn_256',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_relpos_type == "vit_relpos_base_patch16_plus_240":
        try:
            vision_transformer_relpos_model = create_model('vit_relpos_base_patch16_plus_240',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except Exception:
            vision_transformer_relpos_model = create_model('vit_relpos_base_patch16_plus_240',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_relpos_type == "vit_relpos_small_patch16_224":
        try:
            vision_transformer_relpos_model = create_model('vit_relpos_small_patch16_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except Exception:
            vision_transformer_relpos_model = create_model('vit_relpos_small_patch16_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_relpos_type == "vit_relpos_medium_patch16_224":
        try:
            vision_transformer_relpos_model = create_model('vit_relpos_medium_patch16_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except Exception:
            vision_transformer_relpos_model = create_model('vit_relpos_medium_patch16_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_relpos_type == "vit_relpos_base_patch16_224":
        try:
            vision_transformer_relpos_model = create_model('vit_relpos_base_patch16_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except Exception:
            vision_transformer_relpos_model = create_model('vit_relpos_base_patch16_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_relpos_type == "vit_srelpos_small_patch16_224":
        try:
            vision_transformer_relpos_model = create_model('vit_srelpos_small_patch16_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except Exception:
            vision_transformer_relpos_model = create_model('vit_srelpos_small_patch16_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_relpos_type == "vit_srelpos_medium_patch16_224":
        try:
            vision_transformer_relpos_model = create_model('vit_srelpos_medium_patch16_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except Exception:
            vision_transformer_relpos_model = create_model('vit_srelpos_medium_patch16_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_relpos_type == "vit_relpos_medium_patch16_cls_224":
        try:
            vision_transformer_relpos_model = create_model('vit_relpos_medium_patch16_cls_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except Exception:
            vision_transformer_relpos_model = create_model('vit_relpos_medium_patch16_cls_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_relpos_type == "vit_relpos_base_patch16_cls_224":
        try:
            vision_transformer_relpos_model = create_model('vit_relpos_base_patch16_cls_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except Exception:
            vision_transformer_relpos_model = create_model('vit_relpos_base_patch16_cls_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_relpos_type == "vit_relpos_base_patch16_clsgap_224":
        try:
            vision_transformer_relpos_model = create_model('vit_relpos_base_patch16_clsgap_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except Exception:
            vision_transformer_relpos_model = create_model('vit_relpos_base_patch16_clsgap_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_relpos_type == "vit_relpos_small_patch16_rpn_224":
        try:
            vision_transformer_relpos_model = create_model('vit_relpos_small_patch16_rpn_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except Exception:
            vision_transformer_relpos_model = create_model('vit_relpos_small_patch16_rpn_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_relpos_type == "vit_relpos_medium_patch16_rpn_224":
        try:
            vision_transformer_relpos_model = create_model('vit_relpos_medium_patch16_rpn_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except Exception:
            vision_transformer_relpos_model = create_model('vit_relpos_medium_patch16_rpn_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_relpos_type == "vit_relpos_base_patch16_rpn_224":
        try:
            vision_transformer_relpos_model = create_model('vit_relpos_base_patch16_rpn_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except Exception:
            vision_transformer_relpos_model = create_model('vit_relpos_base_patch16_rpn_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Vision Transformer Hybrid Architecture: {vision_transformer_relpos_type}')

    return vision_transformer_relpos_model
