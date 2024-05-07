from timm import create_model


def get_swin_transformer_v2_model(swin_type, num_classes):
    """
     Get a Swin Transformer v2 model for classification/regression.

     Args:
         swin_type (str): Type of Swin Transformer v2 model. Options include:
             - 'swinv2_tiny_window16_256'
             - 'swinv2_tiny_window8_256'
             - 'swinv2_small_window16_256'
             - 'swinv2_small_window8_256'
             - 'swinv2_base_window16_256'
             - 'swinv2_base_window8_256'
             - 'swinv2_base_window12_192'
             - 'swinv2_base_window12to16_192to256'
             - 'swinv2_base_window12to24_192to384'
             - 'swinv2_large_window12_192'
             - 'swinv2_large_window12to16_192to256'
             - 'swinv2_large_window12to24_192to384'
         num_classes (int): Number of output classes.

     Returns:
         torch.nn.Module: Swin Transformer v2 model.
     """
    if swin_type == "swinv2_tiny_window16_256":
        try:
            swin_model = create_model('swinv2_tiny_window16_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            swin_model = create_model('swinv2_tiny_window16_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_tiny_window8_256":
        try:
            swin_model = create_model('swinv2_tiny_window8_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            swin_model = create_model('swinv2_tiny_window8_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_small_window16_256":
        try:
            swin_model = create_model('swinv2_small_window16_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            swin_model = create_model('swinv2_small_window16_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_small_window8_256":
        try:
            swin_model = create_model('swinv2_small_window8_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            swin_model = create_model('swinv2_small_window8_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_base_window16_256":
        try:
            swin_model = create_model('swinv2_base_window16_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            swin_model = create_model('swinv2_base_window16_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_base_window8_256":
        try:
            swin_model = create_model('swinv2_base_window8_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            swin_model = create_model('swinv2_base_window8_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_base_window12_192":
        try:
            swin_model = create_model('swinv2_base_window12_192',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            swin_model = create_model('swinv2_base_window12_192',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_base_window12to16_192to256":
        try:
            swin_model = create_model('swinv2_base_window12to16_192to256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            swin_model = create_model('swinv2_base_window12to16_192to256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_base_window12to24_192to384":
        try:
            swin_model = create_model('swinv2_base_window12to24_192to384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            swin_model = create_model('swinv2_base_window12to24_192to384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_large_window12_192":
        try:
            swin_model = create_model('swinv2_large_window12_192',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            swin_model = create_model('swinv2_large_window12_192',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_large_window12to16_192to256":
        try:
            swin_model = create_model('swinv2_large_window12to16_192to256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            swin_model = create_model('swinv2_large_window12to16_192to256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_large_window12to24_192to384":
        try:
            swin_model = create_model('swinv2_large_window12to24_192to384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except Exception:
            swin_model = create_model('swinv2_large_window12to24_192to384',
                                      pretrained=False,
                                      num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Swin Transformer v2 Architecture : {swin_type}')

    return swin_model
