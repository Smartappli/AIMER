from timm import create_model


def get_swin_transformer_v2_cr_model(swin_type, num_classes):
    if swin_type == "swinv2_cr_tiny_384":
        try:
            swin_model = create_model('swinv2_cr_tiny_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            swin_model = create_model('swinv2_cr_tiny_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_cr_tiny_224":
        try:
            swin_model = create_model('swinv2_cr_tiny_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            swin_model = create_model('swinv2_cr_tiny_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_cr_tiny_ns_224":
        try:
            swin_model = create_model('swinv2_cr_tiny_ns_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            swin_model = create_model('swinv2_cr_tiny_ns_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_cr_small_384":
        try:
            swin_model = create_model('swinv2_cr_small_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            swin_model = create_model('swinv2_cr_small_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_cr_small_224":
        try:
            swin_model = create_model('swinv2_cr_small_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            swin_model = create_model('swinv2_cr_small_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_cr_small_ns_224":
        try:
            swin_model = create_model('swinv2_cr_small_ns_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            swin_model = create_model('swinv2_cr_small_ns_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_cr_small_ns_256":
        try:
            swin_model = create_model('swinv2_cr_small_ns_256',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            swin_model = create_model('swinv2_cr_small_ns_256',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_cr_base_384":
        try:
            swin_model = create_model('swinv2_cr_base_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            swin_model = create_model('swinv2_cr_base_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_cr_base_224":
        try:
            swin_model = create_model('swinv2_cr_base_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            swin_model = create_model('swinv2_cr_base_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_cr_base_ns_224":
        try:
            swin_model = create_model('swinv2_cr_base_ns_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            swin_model = create_model('swinv2_cr_base_ns_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_cr_large_384":
        try:
            swin_model = create_model('swinv2_cr_large_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            swin_model = create_model('swinv2_cr_large_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_cr_large_224":
        try:
            swin_model = create_model('swinv2_cr_large_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            swin_model = create_model('swinv2_cr_large_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_cr_huge_384":
        try:
            swin_model = create_model('swinv2_cr_huge_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            swin_model = create_model('swinv2_cr_huge_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_cr_huge_224":
        try:
            swin_model = create_model('swinv2_cr_huge_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            swin_model = create_model('swinv2_cr_huge_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "swinv2_cr_giant_384":
        try:
            swin_model = create_model('swinv2_cr_giant_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            swin_model = create_model('swinv2_cr_giant_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif swin_type == "winv2_cr_giant_224":
        try:
            swin_model = create_model('winv2_cr_giant_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except:
            swin_model = create_model('winv2_cr_giant_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Swin Transformer v2 cr Architecture : {swin_type}')

    return swin_model