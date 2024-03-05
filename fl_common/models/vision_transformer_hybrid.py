from timm import create_model


def get_vision_transformer_hybrid_model(vision_transformer_hybrid_type, num_classes):
    if vision_transformer_hybrid_type == 'vit_tiny_r_s16_p8_224':
        try:
            vision_transformer_hybrid_model = create_model('vit_tiny_r_s16_p8_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except:
            vision_transformer_hybrid_model = create_model('vit_tiny_r_s16_p8_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_hybrid_type == 'vit_tiny_r_s16_p8_384':
        try:
            vision_transformer_hybrid_model = create_model('vit_tiny_r_s16_p8_384',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except:
            vision_transformer_hybrid_model = create_model('vit_tiny_r_s16_p8_384',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_hybrid_type == 'vit_small_r26_s32_224':
        try:
            vision_transformer_hybrid_model = create_model('vit_small_r26_s32_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except:
            vision_transformer_hybrid_model = create_model('vit_small_r26_s32_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_hybrid_type == 'vit_small_r26_s32_384':
        try:
            vision_transformer_hybrid_model = create_model('vit_small_r26_s32_384',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except:
            vision_transformer_hybrid_model = create_model('vit_small_r26_s32_384',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_hybrid_type == 'vit_base_r26_s32_224':
        try:
            vision_transformer_hybrid_model = create_model('vit_base_r26_s32_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except:
            vision_transformer_hybrid_model = create_model('vit_base_r26_s32_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_hybrid_type == 'vit_base_r50_s16_224':
        try:
            vision_transformer_hybrid_model = create_model('vit_base_r50_s16_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except:
            vision_transformer_hybrid_model = create_model('vit_base_r50_s16_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_hybrid_type == 'vit_base_r50_s16_384':
        try:
            vision_transformer_hybrid_model = create_model('vit_base_r50_s16_384',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except:
            vision_transformer_hybrid_model = create_model('vit_base_r50_s16_384',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_hybrid_type == 'vit_large_r50_s32_224':
        try:
            vision_transformer_hybrid_model = create_model('vit_large_r50_s32_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except:
            vision_transformer_hybrid_model = create_model('vit_large_r50_s32_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_hybrid_type == 'vit_large_r50_s32_384':
        try:
            vision_transformer_hybrid_model = create_model('vit_large_r50_s32_384',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except:
            vision_transformer_hybrid_model = create_model('vit_large_r50_s32_384',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_hybrid_type == 'vit_small_resnet26d_224':
        try:
            vision_transformer_hybrid_model = create_model('vit_small_resnet26d_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except:
            vision_transformer_hybrid_model = create_model('vit_small_resnet26d_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_hybrid_type == 'vit_small_resnet50d_s16_224':
        try:
            vision_transformer_hybrid_model = create_model('vit_small_resnet50d_s16_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except:
            vision_transformer_hybrid_model = create_model('vit_small_resnet50d_s16_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_hybrid_type == 'vit_base_resnet26d_224':
        try:
            vision_transformer_hybrid_model = create_model('vit_base_resnet26d_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except:
            vision_transformer_hybrid_model = create_model('vit_base_resnet26d_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    elif vision_transformer_hybrid_type == 'vit_base_resnet50d_224':
        try:
            vision_transformer_hybrid_model = create_model('vit_base_resnet50d_224',
                                                           pretrained=True,
                                                           num_classes=num_classes)
        except:
            vision_transformer_hybrid_model = create_model('vit_base_resnet50d_224',
                                                           pretrained=False,
                                                           num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Vision Transformer Hybrid Architecture: {vision_transformer_hybrid_type}')

    return vision_transformer_hybrid_model
