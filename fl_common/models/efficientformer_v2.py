from timm import create_model


def get_efficientformer_v2_model(efficientformer_v2_type, num_classes):
    if efficientformer_v2_type == 'efficientformerv2_s0':
        try:
            efficientformer_v2_model = create_model('efficientformerv2_s0',
                                                    pretrained=True,
                                                    num_classes=num_classes)
        except:
            efficientformer_v2_model = create_model('efficientformerv2_s0',
                                                    pretrained=False,
                                                    num_classes=num_classes)
    elif efficientformer_v2_type == 'efficientformerv2_s1':
        try:
            efficientformer_v2_model = create_model('efficientformerv2_s1',
                                                    pretrained=True,
                                                    num_classes=num_classes)
        except:
            efficientformer_v2_model = create_model('efficientformerv2_s1',
                                                    pretrained=False,
                                                    num_classes=num_classes)
    elif efficientformer_v2_type == 'efficientformerv2_s2':
        try:
            efficientformer_v2_model = create_model('efficientformerv2_s2',
                                                    pretrained=True,
                                                    num_classes=num_classes)
        except:
            efficientformer_v2_model = create_model('efficientformerv2_s2',
                                                    pretrained=False,
                                                    num_classes=num_classes)
    elif efficientformer_v2_type == 'efficientformerv2_l':
        try:
            efficientformer_v2_model = create_model('efficientformerv2_l',
                                                    pretrained=True,
                                                    num_classes=num_classes)
        except:
            efficientformer_v2_model = create_model('efficientformerv2_l',
                                                    pretrained=False,
                                                    num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Efficientformer v2 Architecture: {efficientformer_v2_type}')

    return efficientformer_v2_model
