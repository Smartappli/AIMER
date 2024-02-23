import os

from fl_common.models.alexnet import get_alexnet_model
from fl_common.models.beit import get_beit_model
from fl_common.models.cait import get_cait_model
from fl_common.models.coat import get_coat_model
from fl_common.models.convit import get_convit_model
from fl_common.models.convmixer import get_convmixer_model
from fl_common.models.convnext import get_convnext_model
from fl_common.models.crossvit import get_crossvit_model
from fl_common.models.cspnet import get_cspnet_model
from fl_common.models.davit import get_davit_model
from fl_common.models.deit import get_deit_model
from fl_common.models.densenet import get_densenet_model
from fl_common.models.dla import get_dla_model
from fl_common.models.dpn import get_dpn_model
from fl_common.models.edgenet import get_edgenet_model
from fl_common.models.efficientnet import get_efficientnet_model
from fl_common.models.eva import get_eva_model
from fl_common.models.fastvit import get_fastvit_model
from fl_common.models.focalnet import get_focalnet_model
from fl_common.models.gcvit import get_gcvit_model
from fl_common.models.ghostnet import get_ghostnet_model
from fl_common.models.googlenet import get_googlenet_model
from fl_common.models.hardcorenas import get_hardcorenas_model
from fl_common.models.hgnet import get_hgnet_model
from fl_common.models.hrnet import get_hrnet_model
from fl_common.models.inception_next import get_inception_next_model
from fl_common.models.inception import get_inception_model
from fl_common.models.levit import get_levit_model
from fl_common.models.maxvit import get_maxvit_model
from fl_common.models.mnasnet import get_mnasnet_model
from fl_common.models.mobilenet import get_mobilenet_model
from fl_common.models.regnet import get_regnet_model
from fl_common.models.resnet import get_resnet_model
from fl_common.models.resnext import get_resnext_model
from fl_common.models.squeezenet import get_squeezenet_model
from fl_common.models.shufflenet import get_shufflenet_model
from fl_common.models.swin_transformer import get_swin_model
from fl_common.models.tiny_vit import get_tiny_vit_model
from fl_common.models.twins import get_twins_model
from fl_common.models.tresnet import get_tresnet_model
from fl_common.models.vgg import get_vgg_model
from fl_common.models.vision_transformer import get_vision_model
from fl_common.models.volo import get_volo_model
from fl_common.models.vovnet import get_vovnet_model
from fl_common.models.wide_resnet import get_wide_resnet_model
from fl_common.models.xception import get_xception_model
from fl_common.models.xcit import get_xcit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


def get_family_model_a(model_type, num_classes):
    """
    Retrieves a model belonging to family A based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should be one of ['AlexNet'] for family A.
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model if available, otherwise 'Unknown'.
    """
    model = "Unknown"

    if model_type in ['AlexNet']:
        model = get_alexnet_model(alexnet_type='AlexNet', num_classes=num_classes)

    return model


def get_family_model_b(model_type, num_classes):
    """
    Retrieves a model belonging to family B based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should be one of the supported BEiT models:
                        ['beit_base_patch16_224', 'beit_base_patch16_384', 'beit_large_patch16_224',
                         'beit_large_patch16_384', 'beit_large_patch16_512', 'beitv2_base_patch16_224',
                         'beitv2_large_patch16_224'] for family B.
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model if available, otherwise 'Unknown'.
    """
    model = "Unknown"

    if model_type in ['beit_base_patch16_224', 'beit_base_patch16_384', 'beit_large_patch16_224',
                      'beit_large_patch16_384', 'beit_large_patch16_512', 'beitv2_base_patch16_224',
                      'beitv2_large_patch16_224']:
        model = get_beit_model(model_type, num_classes)

    return model


def get_family_model_c(model_type, num_classes):
    """
    Selects and returns a model from a family of different architectures based on the provided model type.

    Parameters:
        model_type (str): The type of model to retrieve. It can be one of the following:
                          - For CAIT models: ['cait_xxs24_224', 'cait_xxs24_384', 'cait_xxs36_224', 'cait_xxs36_384',
                                               'cait_xs24_384', 'cait_s24_224', 'cait_s24_384', 'cait_s36_224',
                                               'cait_m36_224', 'cait_m48_448']
                          - For COAT models: ['coat_tiny', 'coat_mini', 'coat_small', 'coat_lite_tiny',
                                               'coat_lite_mini', 'coat_lite_small', 'coat_lite_medium',
                                               'coat_lite_medium_384']
                          - For Convmixer models: ['convmixer_1536_20', 'convmixer_768_32',
                                                    'convmixer_1024_20_ks9_p14']
                          - For Convit models: ['convit_tiny', 'convit_small', 'convit_base']
                          - For ConvNeXt models: ['ConvNeXt_Tiny', 'ConvNeXt_Small', 'ConvNeXt_Base',
                                                   'ConvNeXt_Large']
                          - For Crossvit models: ['crossvit_tiny_240', 'rossvit_small_240', 'crossvit_base_240',
                                                   'crossvit_9_240', 'crossvit_15_240', 'crossvit_18_240',
                                                   'crossvit_9_dagger_240', 'rossvit_15_dagger_240',
                                                   'crossvit_15_dagger_408', 'crossvit_18_dagger_240',
                                                   'crossvit_18_dagger_408']
                          - For CSPNet models: ["cspresnet50", "cspresnet50d", "cspresnet50w", "cspresnext50",
                                                "cspdarknet53", "darknet17", "darknet21", "sedarknet21",
                                                "darknet53", "darknetaa53", "cs3darknet_s", "cs3darknet_m",
                                                "cs3darknet_l", "cs3darknet_x", "cs3darknet_focus_s",
                                                "cs3darknet_focus_m", "cs3darknet_focus_l", "cs3darknet_focus_x",
                                                "cs3sedarknet_l", "cs3sedarknet_x", "cs3sedarknet_xdw",
                                                "cs3edgenet_x", "cs3se_edgenet_x"]
        num_classes (int): The number of classes for the final classification layer.

    Returns:
        torch.nn.Module: The selected model instance.

    Raises:
        ValueError: If the specified model type is unknown.
    """
    model = "Unknown"

    if model_type in ['cait_xxs24_224', 'cait_xxs24_384', 'cait_xxs36_224', 'cait_xxs36_384', 'cait_xs24_384',
                      'cait_s24_224', 'cait_s24_384', 'cait_s36_224', 'cait_m36_224', 'cait_m48_448']:
        model = get_cait_model(model_type, num_classes)
    elif model_type in ['coat_tiny', 'coat_mini', 'coat_small', 'coat_lite_tiny', 'coat_lite_mini',
                        'coat_lite_small', 'coat_lite_medium', 'coat_lite_medium_384']:
        model = get_coat_model(model_type, num_classes)
    elif model_type in ['convmixer_1536_20', 'convmixer_768_32', 'convmixer_1024_20_ks9_p14']:
        model = get_convmixer_model(model_type, num_classes)
    elif model_type in ['convit_tiny', 'convit_small', 'convit_base']:
        model = get_convit_model(model_type, num_classes)
    elif model_type in ['ConvNeXt_Tiny', 'ConvNeXt_Small', 'ConvNeXt_Base', 'ConvNeXt_Large']:
        model = get_convnext_model(model_type, num_classes)
    elif model_type in ['crossvit_tiny_240', 'rossvit_small_240', 'crossvit_base_240', 'crossvit_9_240',
                        'crossvit_15_240', 'crossvit_18_240', 'crossvit_9_dagger_240', 'rossvit_15_dagger_240',
                        'crossvit_15_dagger_408', 'crossvit_18_dagger_240', 'crossvit_18_dagger_408']:
        model = get_crossvit_model(model_type, num_classes)
    elif model_type in ['cspresnet50', 'cspresnet50d', 'cspresnet50w', 'cspresnext50', 'cspdarknet53', 'darknet17',
                        'darknet21', 'sedarknet21', 'darknet53', 'darknetaa53', 'cs3darknet_s', 'cs3darknet_m',
                        'cs3darknet_l', 'cs3darknet_x', 'cs3darknet_focus_s', 'cs3darknet_focus_m',
                        'cs3darknet_focus_l', 'cs3darknet_focus_x', 'cs3sedarknet_l', 'cs3sedarknet_x',
                        'cs3sedarknet_xdw', 'cs3edgenet_x', 'cs3se_edgenet_x']:
        model = get_cspnet_model(model_type, num_classes)

    return model


def get_family_model_d(model_type, num_classes):
    """
    Get a model from the family of models including Davit, DeiT, DenseNet, DLA, and DPN.

    Parameters:
        model_type (str): Type of the model. Options include:
            - For Davit: 'davit_tiny', 'davit_small', 'davit_base', 'davit_large', 'davit_huge', 'davit_giant'
            - For DeiT: 'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
                        'deit_base_patch16_384', 'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
                        'deit_base_distilled_patch16_224', 'deit_base_distilled_patch16_384', 'deit3_small_patch16_224',
                        'deit3_small_patch16_384', 'deit3_medium_patch16_224', 'deit3_base_patch16_224',
                        'deit3_base_patch16_384', 'deit3_large_patch16_224', 'deit3_large_patch16_384',
                        'deit3_huge_patch14_224'
            - For DenseNet: 'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201'
            - For DLA: 'dla60_res2net', 'dla60_res2next', 'dla34', 'dla46_c', 'dla46x_c', 'dla60x_c', 'dla60', 'dla60x',
                       'dla102', 'dla102x', 'dla102x2', 'dla169'
            - For DPN: 'dpn48b', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107'
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: The selected model.
    """
    model = "Unknown"

    if model_type in ['davit_tiny', 'davit_small', 'davit_base', 'davit_large', 'davit_huge', 'davit_giant']:
        model = get_davit_model(model_type, num_classes)
    elif model_type in ['deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
                        'deit_base_patch16_384', 'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
                        'deit_base_distilled_patch16_224', 'deit_base_distilled_patch16_384', 'deit3_small_patch16_224',
                        'deit3_small_patch16_384', 'deit3_medium_patch16_224', 'deit3_base_patch16_224',
                        'deit3_base_patch16_384', 'deit3_large_patch16_224', 'deit3_large_patch16_384',
                        'deit3_huge_patch14_224']:
        model = get_deit_model(model_type, num_classes)
    elif model_type in ['DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201']:
        model = get_densenet_model(model_type, num_classes)
    elif model_type in ['dla60_res2net', 'dla60_res2next', 'dla34', 'dla46_c', 'dla46x_c', 'dla60x_c', 'dla60',
                        'dla60x', 'dla102', 'dla102x', 'dla102x2', 'dla169']:
        model = get_dla_model(model_type, num_classes)
    elif model_type in ['dpn48b', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107']:
        model = get_dpn_model(model_type, num_classes)

    return model


def get_family_model_e(model_type, num_classes):
    """
    Get a model from the family of models including EdgeNet, EfficientNet, and Eva.

    Parameters:
        model_type (str): Type of the model. Options include:
            - For EdgeNet: 'edgenext_xx_small', 'edgenext_x_small', 'edgenext_small', 'edgenext_base',
                           'edgenext_small_rw'
            - For EfficientNet: 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                                'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7',
                                'EfficientNetV2S', 'EfficientNetV2M', 'EfficientNetV2L'
            - For Eva: 'eva_giant_patch14_224', 'eva_giant_patch14_336', 'eva_giant_patch14_560',
                       'eva02_tiny_patch14_224', 'eva02_small_patch14_224', 'eva02_base_patch14_224',
                       'eva02_large_patch14_224', 'eva02_tiny_patch14_336', 'eva02_small_patch14_336',
                       'eva02_base_patch14_448', 'eva02_large_patch14_448', 'eva_giant_patch14_clip_224',
                       'eva02_base_patch16_clip_224', 'eva02_large_patch14_clip_224', 'eva02_large_patch14_clip_336',
                       'eva02_enormous_patch14_clip_224'
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: The selected model.
    """
    model = "Unknown"

    if model_type in ['edgenext_xx_small', 'edgenext_x_small', 'edgenext_small', 'edgenext_base',
                      'edgenext_small_rw']:
        model = get_edgenet_model(model_type, num_classes)
    elif model_type in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
                        'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2S', 'EfficientNetV2M',
                        'EfficientNetV2L']:
        model = get_efficientnet_model(model_type, num_classes)
    elif model_type in ['eva_giant_patch14_224', 'eva_giant_patch14_336', 'eva_giant_patch14_560',
                        'eva02_tiny_patch14_224', 'eva02_small_patch14_224', 'eva02_base_patch14_224',
                        'eva02_large_patch14_224', 'eva02_tiny_patch14_336', 'eva02_small_patch14_336',
                        'eva02_base_patch14_448', 'eva02_large_patch14_448', 'eva_giant_patch14_clip_224',
                        'eva02_base_patch16_clip_224', 'eva02_large_patch14_clip_224', 'eva02_large_patch14_clip_336',
                        'eva02_enormous_patch14_clip_224']:
        model = get_eva_model(model_type, num_classes)

    return model


def get_family_model_f(model_type, num_classes):
    """
    Get a model from the family of models including FastViT and FocalNet.

    Parameters:
        model_type (str): Type of the model. Options include:
            - For FastViT: 'fastvit_t8', 'fastvit_t12', 'fastvit_s12', 'fastvit_sa12', 'fastvit_sa24', 'fastvit_sa36',
                           'fastvit_ma36'
            - For FocalNet: 'focalnet_tiny_srf', 'focalnet_small_srf', 'focalnet_base_srf', 'focalnet_tiny_lrf',
                            'focalnet_small_lrf', 'focalnet_base_lrf', 'focalnet_large_fl3', 'focalnet_large_fl4',
                            'focalnet_xlarge_fl3', 'focalnet_xlarge_fl4', 'focalnet_huge_fl3', 'focalnet_huge_fl4'
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: The selected model.
    """
    model = "Unknown"

    if model_type in ['fastvit_t8', 'fastvit_t12', 'fastvit_s12', 'fastvit_sa12', 'fastvit_sa24', 'fastvit_sa36',
                      'fastvit_ma36']:
        model = get_fastvit_model(model_type, num_classes)
    elif model_type in ['focalnet_tiny_srf', 'focalnet_small_srf', 'focalnet_base_srf', 'focalnet_tiny_lrf',
                        'focalnet_small_lrf', 'focalnet_base_lrf', 'focalnet_large_fl3', 'focalnet_large_fl4',
                        'focalnet_xlarge_fl3', 'focalnet_xlarge_fl4', 'focalnet_huge_fl3', 'focalnet_huge_fl4']:
        model = get_focalnet_model(model_type, num_classes)
    return model


def get_family_model_g(model_type, num_classes):
    """
    Get a model from the family of models including GCViT, GhostNet, and GoogLeNet.

    Parameters:
        model_type (str): Type of the model. Options include:
            - For GCViT: 'gcvit_xxtiny', 'gcvit_xtiny', 'gcvit_tiny', 'gcvit_small', 'gcvit_base'
            - For GhostNet: 'ghostnet_050', 'ghostnet_100', 'ghostnet_130', 'ghostnetv2_100', 'ghostnetv2_130',
                            'ghostnetv2_160'
            - For GoogLeNet: 'GoogLeNet'
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: The selected model.
    """
    model = "Unknown"

    if model_type in ['gcvit_xxtiny', 'gcvit_xtiny', 'gcvit_tiny', 'gcvit_small', 'gcvit_base']:
        model = get_gcvit_model(model_type, num_classes)
    elif model_type in ['ghostnet_050', 'ghostnet_100', 'gostnet_130', 'ghostnetv2_100', 'ghostnetv2_130',
                        'ghostnetv2_160']:
        model = get_ghostnet_model(model_type, num_classes)
    elif model_type == 'GoogLeNet':
        model = get_googlenet_model(model_type, num_classes)
    return model


def get_family_model_h(model_type, num_classes):
    """
    Get a model from the family of models including HardcoreNAS, HGNet, and HRNet.

    Parameters:
        model_type (str): Type of the model. Options include:
            - For HardcoreNAS: 'hardcorenas_a', 'hardcorenas_b', 'hardcorenas_c', 'hardcorenas_d', 'hardcorenas_e',
                               'hardcorenas_f'
            - For HGNet: 'hgnet_tiny', 'hgnet_small', 'hgnet_base', 'hgnetv2_b0', 'hgnetv2_b1', 'hgnetv2_b2',
                         'hgnetv2_b3', 'hgnetv2_b4', 'hgnetv2_b5', 'hgnetv2_b6'
            - For HRNet: 'hrnet_w18_small', 'hrnet_w18_small_v2', 'hrnet_w18', 'hrnet_w30', 'hrnet_w32', 'hrnet_w40',
                         'hrnet_w44', 'hrnet_w48', 'hrnet_w64', 'hrnet_w18_ssld', 'hrnet_w48_ssld'
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: The selected model.
    """
    model = "Unknown"

    if model_type in ['hardcorenas_a', 'hardcorenas_b', 'hardcorenas_c', 'hardcorenas_d', 'hardcorenas_e',
                      'hardcorenas_f']:
        model = get_hardcorenas_model(model_type, num_classes)
    elif model_type in ['hgnet_tiny', 'hgnet_small', 'hgnet_base', 'hgnetv2_b0', 'hgnetv2_b1', 'hgnetv2_b2',
                        'hgnetv2_b3', 'hgnetv2_b4', 'hgnetv2_b5', 'hgnetv2_b6']:
        model = get_hgnet_model(model_type, num_classes)
    elif model_type in ['hrnet_w18_small', 'hrnet_w18_small_v2', 'hrnet_w18', 'hrnet_w30', 'hrnet_w32', 'hrnet_w40',
                        'hrnet_w44', 'hrnet_w48', 'hrnet_w64', 'hrnet_w18_ssld', 'hrnet_w48_ssld']:
        model = get_hrnet_model(model_type, num_classes)
    return model


def get_family_model_i(model_type, num_classes):
    """
    Retrieves a model belonging to family I based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should be one of the supported models:
                        - For Inception models: ['Inception_V3', 'inception_v4', 'inception_resnet_v2']
                        - For InceptionNext models: ['inception_next_tiny', 'inception_next_small',
                                                      'inception_next_base']
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model if available, otherwise 'Unknown'.
    """
    model = "Unknown"

    if model_type in ['Inception_V3', 'inception_v4', 'inception_resnet_v2']:
        model = get_inception_model(model_type, num_classes)
    elif model_type in ['inception_next_tiny', 'inception_next_small', 'inception_next_base']:
        model = get_inception_next_model(model_type, num_classes)

    return model


def get_family_model_l(model_type, num_classes):
    """
    Retrieves a model belonging to family L based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should be one of the supported models:
                        - For LeViT models: ['levit_128s', 'levit_128', 'levit_192', 'levit_256', 'levit_384',
                                             'levit_384_s8', 'levit_512_s8', 'levit_512', 'levit_256d', 'levit_512d',
                                             'levit_conv_128s', 'levit_conv_128', 'levit_conv_192', 'levit_conv_256',
                                             'levit_conv_384', 'levit_conv_384_s8', 'levit_conv_512_s8', 'levit_conv_512',
                                             'levit_conv_256d', 'levit_conv_512d']
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model if available, otherwise 'Unknown'.
    """
    model = "Unknown"

    if model_type in ['levit_128s', 'levit_128', 'levit_192', 'levit_256', 'levit_384', 'levit_384_s8',
                      'levit_512_s8', 'levit_512', 'levit_256d', 'levit_512d', 'levit_conv_128s',
                      'levit_conv_128', 'levit_conv_192', 'levit_conv_256', 'levit_conv_384', 'levit_conv_384_s8',
                      'levit_conv_512_s8', 'levit_conv_512', 'levit_conv_256d', 'levit_conv_512d']:
        model = get_levit_model(model_type, num_classes)

    return model


def get_family_model_m(model_type, num_classes):
    """
    Retrieves a model belonging to family M based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should be one of the supported models:
                        - For MaxViT models: ['MaxVit_T']
                        - For MNASNet models: ['MNASNet0_5', 'MNASNet0_75', 'MNASNet1_0', 'MNASNet1_3']
                        - For MobileNet models: ['MobileNet_V2', 'MobileNet_V3_Small', 'MobileNet_V3_Large']
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model if available, otherwise 'Unknown'.
    """
    model = "Unknown"

    if model_type in ['MaxVit_T']:
        model = get_maxvit_model(model_type, num_classes)
    elif model_type in ['MNASNet0_5', 'MNASNet0_75', 'MNASNet1_0', 'MNASNet1_3']:
        model = get_mnasnet_model(model_type, num_classes)
    elif model_type in ['MobileNet_V2', 'MobileNet_V3_Small', 'MobileNet_V3_Large']:
        model = get_mobilenet_model(model_type, num_classes)

    return model


def get_family_model_r(model_type, num_classes):
    """
    Retrieves a model belonging to family R based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should be one of the supported models:
                        - For RegNet models: ['RegNet_X_400MF', 'RegNet_X_800MF', 'RegNet_X_1_6GF',
                                              'RegNet_X_3_2GF', 'RegNet_X_16GF', 'RegNet_Y_400MF', 'RegNet_Y_800MF',
                                              'RegNet_Y_1_6GF', 'RegNet_Y_3_2GF', 'RegNet_Y_16GF']
                        - For ResNet models: ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                                              'ResNeXt50_32X4D', 'ResNeXt101_32X4D', 'ResNeXt101_64X4D']
                        - For ResNeXt models: ['ResNeXt50_32X4D', 'ResNeXt101_32X8D', 'ResNeXt101_64X4D']
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model if available, otherwise 'Unknown'.
    """
    model = "Unknown"

    if model_type in ['RegNet_X_400MF', 'RegNet_X_800MF', 'RegNet_X_1_6GF', 'RegNet_X_3_2GF', 'RegNet_X_16GF',
                      'RegNet_Y_400MF', 'RegNet_Y_800MF', 'RegNet_Y_1_6GF', 'RegNet_Y_3_2GF', 'RegNet_Y_16GF']:
        model = get_regnet_model(model_type, num_classes)
    elif model_type in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNeXt50_32X4D',
                        'ResNeXt101_32X4D', 'ResNeXt101_64X4D']:
        model = get_resnet_model(model_type, num_classes)
    elif model_type in ['ResNeXt50_32X4D', 'ResNeXt101_32X8D', 'ResNeXt101_64X4D']:
        model = get_resnext_model(model_type, num_classes)

    return model


def get_family_model_s(model_type, num_classes):
    """
    Retrieves a model belonging to family S based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should be one of the supported models:
                        - For ShuffleNet V2 models: ['ShuffleNet_V2_X0_5', 'ShuffleNet_V2_X1_0',
                                                      'ShuffleNet_V2_X1_5', 'ShuffleNet_V2_X2_0']
                        - For SqueezeNet models: ['SqueezeNet1_0', 'SqueezeNet1_1']
                        - For Swin Transformer models: ['Swin_T', 'Swin_S', 'Swin_B', 'Swin_V2_T', 'Swin_V2_S',
                                                         'Swin_V2_B']
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model if available, otherwise 'Unknown'.
    """
    model = "Unknown"

    if model_type in ['ShuffleNet_V2_X0_5', 'ShuffleNet_V2_X1_0', 'ShuffleNet_V2_X1_5', 'ShuffleNet_V2_X2_0']:
        model = get_shufflenet_model(model_type, num_classes)
    elif model_type in ["SqueezeNet1_0", 'SqueezeNet1_1']:
        model = get_squeezenet_model(model_type, num_classes)
    elif model_type in ['Swin_T', 'Swin_S', 'Swin_B', 'Swin_V2_T', 'Swin_V2_S', 'Swin_V2_B']:
        model = get_swin_model(model_type, num_classes)

    return model


def get_family_model_t(model_type, num_classes):
    """
    Get a model from the family of models including TinyViT, TResNet, and Twins.

    Parameters:
        model_type (str): Type of the model. Options include:
            - For TinyViT: 'tiny_vit_5m_224', 'tiny_vit_11m_224', 'tiny_vit_21m_224', 'tiny_vit_21m_384', 'tiny_vit_21m_512'
            - For TResNet: 'tresnet_m', 'tresnet_l', 'tresnet_xl', 'tresnet_v2_l'
            - For Twins: 'twins_pcpvt_small', 'twins_pcpvt_base', 'twins_pcpvt_large', 'twins_svt_small',
                         'twins_svt_base', 'twins_svt_large'
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: The selected model.

    """
    model = "Unknown"

    if model_type in ['tiny_vit_5m_224', 'tiny_vit_11m_224', 'tiny_vit_21m_224', 'tiny_vit_21m_384',
                      'tiny_vit_21m_512']:
        model = get_tiny_vit_model(model_type, num_classes)
    elif model_type in ['tresnet_m', 'tresnet_l', 'tresnet_xl', 'tresnet_v2_l']:
        model = get_tresnet_model(model_type, num_classes)
    elif model_type in ['twins_pcpvt_small', 'twins_pcpvt_base', 'twins_pcpvt_large', 'twins_svt_small',
                        'twins_svt_base', 'twins_svt_large']:
        model = get_twins_model(model_type, num_classes)
    return model


def get_family_model_v(model_type, num_classes):
    """
    Retrieves a model belonging to family V based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should be one of the supported models:
                        - For VGG models: ['VGG11', 'VGG11_BN', 'VGG13', 'VGG13_BN', 'VGG16', 'VGG16_BN', 'VGG19',
                                          'VGG19_BN']
                        - For Vision Transformer models: ['ViT_B_16', 'ViT_B_32', 'ViT_L_16', 'ViT_L_32', 'ViT_H_14']
                        - For VOLO models: ['volo_d1_224', 'volo_d1_384', 'volo_d2_224', 'volo_d2_384', 'volo_d3_224',
                                             'volo_d3_448', 'volo_d4_224', 'volo_d4_448', 'volo_d5_224', 'volo_d5_448',
                                             'volo_d5_512']
                        - For VOvNet models: ['vovnet39a', 'vovnet57a', 'ese_vovnet19b_slim_dw', 'ese_vovnet19b_slim_dw',
                                               'ese_vovnet19b_slim', 'ese_vovnet39b', 'ese_vovnet57b', 'ese_vovnet99b',
                                               'eca_vovnet39b', 'eca_vovnet39b_evos']
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model if available, otherwise 'Unknown'.
    """
    model = "Unknown"

    if model_type in ['VGG11', 'VGG11_BN', 'VGG13', 'VGG13_BN', 'VGG16', 'VGG16_BN', 'VGG19', 'VGG19_BN']:
        model = get_vgg_model(model_type, num_classes)
    elif model_type in ['ViT_B_16', 'ViT_B_32', 'ViT_L_16', 'ViT_L_32', 'ViT_H_14']:
        model = get_vision_model(model_type, num_classes)
    elif model_type in ['volo_d1_224', 'volo_d1_384', 'volo_d2_224', 'volo_d2_384', 'volo_d3_224', 'volo_d3_448',
                        'volo_d4_224', 'volo_d4_448', 'volo_d5_224', 'volo_d5_448', 'volo_d5_512']:
        model = get_volo_model(model_type, num_classes)
    elif model_type in ['vovnet39a', 'vovnet57a', 'ese_vovnet19b_slim_dw', 'ese_vovnet19b_slim_dw',
                        'ese_vovnet19b_slim', 'ese_vovnet39b', 'ese_vovnet57b', 'ese_vovnet99b',
                        'eca_vovnet39b', 'eca_vovnet39b_evos']:
        model = get_vovnet_model(model_type, num_classes)

    return model


def get_family_model_w(model_type, num_classes):
    """
    Retrieves a model belonging to family W based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should be one of the supported models:
                        - For Wide ResNet models: ['Wide_ResNet50_2', 'Wide_ResNet101_2']
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model if available, otherwise 'Unknown'.
    """
    model = "Unknown"

    if model_type in ['Wide_ResNet50_2', 'Wide_ResNet101_2']:
        model = get_wide_resnet_model(model_type, num_classes)

    return model


def get_family_model_x(model_type, num_classes):
    """
    Retrieves a model belonging to family X based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should be one of the supported models:
                        - For Xception models: ['legacy_xception', 'xception41', 'xception65', 'xception71',
                                                'xception41p', 'xception65p']
                        - For XCiT models: ['xcit_nano_12_p16_224', 'xcit_nano_12_p16_384', 'xcit_tiny_12_p16_224',
                                            'xcit_tiny_12_p16_384', 'xcit_small_12_p16_224', 'xcit_small_12_p16_384',
                                            'xcit_tiny_24_p16_224', 'xcit_tiny_24_p16_384', 'xcit_small_24_p16_224',
                                            'xcit_small_24_p16_384', 'xcit_medium_24_p16_224', 'xcit_medium_24_p16_384',
                                            'xcit_large_24_p16_224', 'xcit_large_24_p16_384', 'xcit_nano_12_p8_224',
                                            'xcit_nano_12_p8_384', 'xcit_tiny_12_p8_224', 'xcit_tiny_12_p8_384',
                                            'xcit_small_12_p8_224', 'xcit_small_12_p8_384', 'xcit_tiny_24_p8_224',
                                            'xcit_tiny_24_p8_384', 'xcit_small_24_p8_224', 'xcit_small_24_p8_384',
                                            'xcit_medium_24_p8_224', 'xcit_medium_24_p8_384', 'xcit_large_24_p8_224',
                                            'xcit_large_24_p8_384']
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model if available, otherwise 'Unknown'.
    """
    model = "Unknown"

    if model_type in ['legacy_xception', 'xception41', 'xception65', 'xception71', 'xception41p', 'xception65p']:
        model = get_xception_model(model_type, num_classes)
    elif model_type in ['xcit_nano_12_p16_224', 'xcit_nano_12_p16_384', 'xcit_tiny_12_p16_224', 'xcit_tiny_12_p16_384',
                        'xcit_small_12_p16_224', 'xcit_small_12_p16_384', 'xcit_tiny_24_p16_224', 'xcit_tiny_24_p16_384',
                        'xcit_small_24_p16_224', 'xcit_small_24_p16_384', 'xcit_medium_24_p16_224',
                        'xcit_medium_24_p16_384', 'xcit_large_24_p16_224', 'xcit_large_24_p16_384', 'xcit_nano_12_p8_224',
                        'xcit_nano_12_p8_384', 'xcit_tiny_12_p8_224', 'xcit_tiny_12_p8_384', 'xcit_small_12_p8_224',
                        'xcit_small_12_p8_384', 'xcit_tiny_24_p8_224', 'xcit_tiny_24_p8_384', 'xcit_small_24_p8_224',
                        'xcit_small_24_p8_384', 'xcit_medium_24_p8_224', 'xcit_medium_24_p8_384', 'xcit_large_24_p8_224',
                        'xcit_large_24_p8_384']:
        model = get_xcit_model(model_type, num_classes)

    return model


def get_family_model(model_type, num_classes):
    """
    Retrieves a model based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should start with a letter indicating the model family.
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model based on the provided model type and number of classes.
    """
    # Convert the first letter of the model_type to lowercase
    first_letter = model_type[0].lower()
    model = "Error"  # Default value if no matching case is found

    # Choose the appropriate model based on the first letter of model_type
    match first_letter:
        case 'a':
            model = get_family_model_a(model_type, num_classes)
        case 'b':
            model = get_family_model_b(model_type, num_classes)
        case 'c':
            model = get_family_model_c(model_type, num_classes)
        case 'e':
            model = get_family_model_e(model_type, num_classes)
        case 'f':
            model = get_family_model_f(model_type, num_classes)
        case 'g':
            model = get_family_model_g(model_type, num_classes)
        case 'h':
            model = get_family_model_h(model_type, num_classes)
        case 'i':
            model = get_family_model_i(model_type, num_classes)
        case 'l':
            model = get_family_model_l(model_type, num_classes)
        case 'm':
            model = get_family_model_m(model_type, num_classes)
        case 'r':
            model = get_family_model_r(model_type, num_classes)
        case 's':
            model = get_family_model_s(model_type, num_classes)
        case 't':
            model = get_family_model_t(model_type, num_classes)
        case 'v':
            model = get_family_model_v(model_type, num_classes)
        case 'w':
            model = get_family_model_w(model_type, num_classes)
        case 'x':
            model = get_family_model_x(model_type, num_classes)

    return model
