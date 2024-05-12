import os

from fl_common.models.alexnet import get_alexnet_model
from fl_common.models.beit import get_beit_model
from fl_common.models.byoanet import get_byoanet_model
from fl_common.models.byobnet import get_byobnet_model
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
from fl_common.models.efficientformer import get_efficientformer_model
from fl_common.models.efficientformer_v2 import get_efficientformer_v2_model
from fl_common.models.efficientvit_mit import get_efficientvit_mit_model
from fl_common.models.efficientvit_msra import get_efficientvit_msra_model
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
from fl_common.models.metaformer import get_metaformer_model
from fl_common.models.mlp_mixer import get_mlp_mixer_model
from fl_common.models.mnasnet import get_mnasnet_model
from fl_common.models.mobilenet import get_mobilenet_model
from fl_common.models.mobilevit import get_mobilevit_model
from fl_common.models.mvitv2 import get_mvitv2_model
from fl_common.models.nasnet import get_nasnet_model
from fl_common.models.nest import get_nest_model
from fl_common.models.nextvit import get_nextvit_model
from fl_common.models.nfnet import get_nfnet_model
from fl_common.models.pit import get_pit_model
from fl_common.models.pnasnet import get_pnasnet_model
from fl_common.models.pvt_v2 import get_pvt_v2_model
from fl_common.models.regnet import get_regnet_model
from fl_common.models.repghost import get_repghost_model
from fl_common.models.repvit import get_repvit_model
from fl_common.models.res2net import get_res2net_model
from fl_common.models.resnest import get_resnest_model
from fl_common.models.resnetv2 import get_resnetv2_model
from fl_common.models.resnet import get_resnet_model
from fl_common.models.resnext import get_resnext_model
from fl_common.models.rexnet import get_rexnet_model
from fl_common.models.selecsls import get_selecsls_model
from fl_common.models.senet import get_senet_model
from fl_common.models.sequencer import get_sequencer_model
from fl_common.models.shufflenet import get_shufflenet_model
from fl_common.models.sknet import get_sknet_model
from fl_common.models.squeezenet import get_squeezenet_model
from fl_common.models.swin_transformer import get_swin_transformer_model
from fl_common.models.swin_transformer_v2 import get_swin_transformer_v2_model
from fl_common.models.swin_transformer_v2_cr import get_swin_transformer_v2_cr_model
from fl_common.models.tiny_vit import get_tiny_vit_model
from fl_common.models.tnt import get_tnt_model
from fl_common.models.twins import get_twins_model
from fl_common.models.tresnet import get_tresnet_model
from fl_common.models.vgg import get_vgg_model
from fl_common.models.visformer import get_visformer_model
from fl_common.models.vision_transformer import get_vision_transformer_model
from fl_common.models.vision_transformer_hybrid import get_vision_transformer_hybrid_model
from fl_common.models.vision_transformer_relpos import get_vision_transformer_relpos_model
from fl_common.models.vision_transformer_sam import get_vision_transformer_sam_model
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
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model = "Unknown"

    if model_type in ['beit_base_patch16_224', 'beit_base_patch16_384', 'beit_large_patch16_224',
                      'beit_large_patch16_384', 'beit_large_patch16_512', 'beitv2_base_patch16_224',
                      'beitv2_large_patch16_224']:
        model = get_beit_model(model_type, num_classes)
    elif model_type in ['botnet26t_256', 'sebotnet33ts_256', 'botnet50ts_256', 'eca_botnext26ts_256', 'halonet_h1',
                        'halonet26t', 'sehalonet33ts', 'halonet50ts', 'eca_halonext26ts', 'lambda_resnet26t',
                        'lambda_resnet50ts', 'lambda_resnet26rpt_256', 'haloregnetz_b', 'lamhalobotnet50ts_256',
                        'halo2botnet50ts_256']:
        model = get_byoanet_model(model_type, num_classes)
    elif model_type in ['gernet_l', 'gernet_m', 'gernet_s', 'repvgg_a0', 'repvgg_a1', 'repvgg_a2', 'repvgg_b0',
                        'repvgg_b1', 'repvgg_b1g4', 'repvgg_b2', 'repvgg_b2g4', 'repvgg_b3', 'repvgg_b3g4',
                        'repvgg_d2se', 'resnet51q', 'resnet61q', 'resnext26ts', 'gcresnext26ts', 'seresnext26ts',
                        'eca_resnext26ts', 'bat_resnext26ts', 'resnet32ts', 'resnet33ts', 'gcresnet33ts',
                        'seresnet33ts', 'eca_resnet33ts', 'gcresnet50t', 'gcresnext50ts', 'regnetz_b16',
                        'regnetz_c16', 'regnetz_d32', 'regnetz_d8', 'regnetz_e8', 'regnetz_b16_evos',
                        'regnetz_c16_evos', 'regnetz_d8_evos', 'mobileone_s0', 'mobileone_s1', 'mobileone_s2',
                        'mobileone_s3', "mobileone_s4"]:
        model = get_byobnet_model(model_type, num_classes)

    return model


def get_family_model_c(model_type, num_classes):
    """
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model_retrieval_functions = {
        'cait_xxs24_224': get_cait_model,
        'cait_xxs24_384': get_cait_model,
        'cait_xxs36_224': get_cait_model,
        'cait_xxs36_384': get_cait_model,
        'cait_xs24_384': get_cait_model,
        'cait_s24_224': get_cait_model,
        'cait_s24_384': get_cait_model,
        'cait_s36_224': get_cait_model,
        'cait_m36_224': get_cait_model,
        'cait_m48_448': get_cait_model,
        'coat_tiny': get_coat_model,
        'coat_mini': get_coat_model,
        'coat_small': get_coat_model,
        'coat_lite_tiny': get_coat_model,
        'coat_lite_mini': get_coat_model,
        'coat_lite_small': get_coat_model,
        'coat_lite_medium': get_coat_model,
        'coat_lite_medium_384': get_coat_model,
        'convmixer_1536_20': get_convmixer_model,
        'convmixer_768_32': get_convmixer_model,
        'convmixer_1024_20_ks9_p14': get_convmixer_model,
        'convit_tiny': get_convit_model,
        'convit_small': get_convit_model,
        'convit_base': get_convit_model,
        'ConvNeXt_Tiny': get_convnext_model,
        'ConvNeXt_Small': get_convnext_model,
        'ConvNeXt_Base': get_convnext_model,
        'ConvNeXt_Large': get_convnext_model,
        'convnext_atto': get_convnext_model,
        'convnext_atto_ols': get_convnext_model,
        'convnext_femto': get_convnext_model,
        'convnext_femto_ols': get_convnext_model,
        'convnext_pico': get_convnext_model,
        'convnext_pico_ols': get_convnext_model,
        'convnext_nano': get_convnext_model,
        'convnext_nano_ols': get_convnext_model,
        'convnext_tiny_hnf': get_convnext_model,
        'convnext_tiny': get_convnext_model,
        'convnext_small': get_convnext_model,
        'convnext_base': get_convnext_model,
        'convnext_large': get_convnext_model,
        'convnext_large_mlp': get_convnext_model,
        'convnext_xlarge': get_convnext_model,
        'convnext_xxlarge': get_convnext_model,
        'convnextv2_atto': get_convnext_model,
        'convnextv2_femto': get_convnext_model,
        'convnextv2_pico': get_convnext_model,
        'convnextv2_nano': get_convnext_model,
        'convnextv2_tiny': get_convnext_model,
        'convnextv2_small': get_convnext_model,
        'convnextv2_base': get_convnext_model,
        'convnextv2_large': get_convnext_model,
        'convnextv2_huge': get_convnext_model,
        'crossvit_tiny_240': get_crossvit_model,
        'rossvit_small_240': get_crossvit_model,
        'crossvit_base_240': get_crossvit_model,
        'crossvit_9_240': get_crossvit_model,
        'crossvit_15_240': get_crossvit_model,
        'crossvit_18_240': get_crossvit_model,
        'crossvit_9_dagger_240': get_crossvit_model,
        'rossvit_15_dagger_240': get_crossvit_model,
        'crossvit_15_dagger_408': get_crossvit_model,
        'crossvit_18_dagger_240': get_crossvit_model,
        'crossvit_18_dagger_408': get_crossvit_model,
        'cspresnet50': get_cspnet_model,
        'cspresnet50d': get_cspnet_model,
        'cspresnet50w': get_cspnet_model,
        'cspresnext50': get_cspnet_model,
        'cspdarknet53': get_cspnet_model,
        'darknet17': get_cspnet_model,
        'darknet21': get_cspnet_model,
        'sedarknet21': get_cspnet_model,
        'darknet53': get_cspnet_model,
        'darknetaa53': get_cspnet_model,
        'cs3darknet_s': get_cspnet_model,
        'cs3darknet_m': get_cspnet_model,
        'cs3darknet_l': get_cspnet_model,
        'cs3darknet_x': get_cspnet_model,
        'cs3darknet_focus_s': get_cspnet_model,
        'cs3darknet_focus_m': get_cspnet_model,
        'cs3darknet_focus_l': get_cspnet_model,
        'cs3darknet_focus_x': get_cspnet_model,
        'cs3sedarknet_l': get_cspnet_model,
        'cs3sedarknet_x': get_cspnet_model,
        'cs3sedarknet_xdw': get_cspnet_model,
        'cs3edgenet_x': get_cspnet_model,
        'cs3se_edgenet_x': get_cspnet_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError("Unknown model_type provided")

    return model_retrieval_functions[model_type](model_type, num_classes)


def get_family_model_d(model_type, num_classes):
    """
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model_retrieval_functions = {
        'davit_tiny': get_davit_model,
        'davit_small': get_davit_model,
        'davit_base': get_davit_model,
        'davit_large': get_davit_model,
        'davit_huge': get_davit_model,
        'davit_giant': get_davit_model,
        'deit_tiny_patch16_224': get_deit_model,
        'deit_small_patch16_224': get_deit_model,
        'deit_base_patch16_224': get_deit_model,
        'deit_base_patch16_384': get_deit_model,
        'deit_tiny_distilled_patch16_224': get_deit_model,
        'deit_small_distilled_patch16_224': get_deit_model,
        'deit_base_distilled_patch16_224': get_deit_model,
        'deit_base_distilled_patch16_384': get_deit_model,
        'deit3_small_patch16_224': get_deit_model,
        'deit3_small_patch16_384': get_deit_model,
        'deit3_medium_patch16_224': get_deit_model,
        'deit3_base_patch16_224': get_deit_model,
        'deit3_base_patch16_384': get_deit_model,
        'deit3_large_patch16_224': get_deit_model,
        'deit3_large_patch16_384': get_deit_model,
        'deit3_huge_patch14_224': get_deit_model,
        'DenseNet121': get_densenet_model,
        'DenseNet161': get_densenet_model,
        'DenseNet169': get_densenet_model,
        'DenseNet201': get_densenet_model,
        'densenet121': get_densenet_model,
        'densenetblur121d': get_densenet_model,
        'densenet169': get_densenet_model,
        'densenet201': get_densenet_model,
        'densenet161': get_densenet_model,
        'densenet264d': get_densenet_model,
        'dla60_res2net': get_dla_model,
        'dla60_res2next': get_dla_model,
        'dla34': get_dla_model,
        'dla46_c': get_dla_model,
        'dla46x_c': get_dla_model,
        'dla60x_c': get_dla_model,
        'dla60': get_dla_model,
        'dla60x': get_dla_model,
        'dla102': get_dla_model,
        'dla102x': get_dla_model,
        'dla102x2': get_dla_model,
        'dla169': get_dla_model,
        'dpn48b': get_dpn_model,
        'dpn68': get_dpn_model,
        'dpn68b': get_dpn_model,
        'dpn92': get_dpn_model,
        'dpn98': get_dpn_model,
        'dpn131': get_dpn_model,
        'dpn107': get_dpn_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError("Unknown model_type provided")

    return model_retrieval_functions[model_type](model_type, num_classes)


def get_family_model_e(model_type, num_classes):
    """
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model = "Unknown"

    if model_type in ['edgenext_xx_small', 'edgenext_x_small', 'edgenext_small', 'edgenext_base',
                      'edgenext_small_rw']:
        model = get_edgenet_model(model_type, num_classes)
    elif model_type in ['efficientformer_l1', 'efficientformer_l3', 'efficientformer_l7']:
        model = get_efficientformer_model(model_type, num_classes)
    elif model_type in ['efficientformerv2_s0', 'efficientformerv2_s1', 'efficientformerv2_s2', 'efficientformerv2_l']:
        model = get_efficientformer_v2_model(model_type, num_classes)
    elif model_type in ['efficientvit_b0', 'efficientvit_b1', 'efficientvit_b2', 'efficientvit_b3', 'efficientvit_l1',
                        'efficientvit_l2', 'efficientvit_l3']:
        model = get_efficientvit_mit_model(model_type, num_classes)
    elif model_type in ['efficientvit_m0', 'efficientvit_m1', 'efficientvit_m2', 'efficientvit_m3', 'efficientvit_m4',
                        'efficientvit_m5']:
        model = get_efficientvit_msra_model(model_type, num_classes)
    elif model_type in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
                        'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2S',
                        'EfficientNetV2M', 'EfficientNetV2L', 'mnasnet_050', 'mnasnet_075', 'mnasnet_100',
                        'mnasnet_140', 'semnasnet_050', 'semnasnet_075', 'semnasnet_100', 'semnasnet_140',
                        'mnasnet_small', 'mobilenetv2_035', 'mobilenetv2_050', 'mobilenetv2_075',
                        'mobilenetv2_100', 'mobilenetv2_140', 'mobilenetv2_110d', 'mobilenetv2_120d', 'fbnetc_100',
                        'spnasnet_100', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                        'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                        'efficientnet_b8', 'efficientnet_l2', 'efficientnet_b0_gn', 'efficientnet_b0_g8_gn',
                        'efficientnet_b0_g16_evos', 'efficientnet_b3_gn', 'efficientnet_b3_g8_gn',
                        'efficientnet_es', 'efficientnet_es_pruned', 'efficientnet_em', 'efficientnet_el',
                        'efficientnet_el_pruned', 'efficientnet_cc_b0_4e', 'efficientnet_cc_b0_8e',
                        'efficientnet_cc_b1_8e', 'efficientnet_lite0', 'efficientnet_lite1', 'efficientnet_lite2',
                        'efficientnet_lite3', 'efficientnet_lite4', 'efficientnet_b1_pruned',
                        'efficientnet_b2_pruned', 'efficientnet_b3_pruned', 'efficientnetv2_rw_t',
                        'gc_efficientnetv2_rw_t', 'efficientnetv2_rw_s', 'efficientnetv2_rw_m', 'efficientnetv2_s',
                        'efficientnetv2_m', 'efficientnetv2_l', 'efficientnetv2_xl', 'tf_efficientnet_b0',
                        'tf_efficientnet_b1', 'tf_efficientnet_b2', 'tf_efficientnet_b3', 'tf_efficientnet_b4',
                        'tf_efficientnet_b5', 'tf_efficientnet_b6', 'tf_efficientnet_b7', 'tf_efficientnet_b8',
                        'tf_efficientnet_l2', 'tf_efficientnet_es', 'tf_efficientnet_em', 'tf_efficientnet_el',
                        'tf_efficientnet_cc_b0_4e', 'tf_efficientnet_cc_b0_8e', 'tf_efficientnet_cc_b1_8e',
                        'tf_efficientnet_lite0', 'tf_efficientnet_lite1', 'tf_efficientnet_lite2',
                        'tf_efficientnet_lite3', 'tf_efficientnet_lite4', 'tf_efficientnetv2_s',
                        'tf_efficientnetv2_m', 'tf_efficientnetv2_l', 'tf_efficientnetv2_xl',
                        'tf_efficientnetv2_b0', 'tf_efficientnetv2_b1', 'tf_efficientnetv2_b2',
                        'tf_efficientnetv2_b3', 'mixnet_s', 'mixnet_m', 'mixnet_l', 'mixnet_xl', 'mixnet_xxl',
                        'tf_mixnet_s', 'tf_mixnet_m', 'tf_mixnet_l', 'tinynet_a', 'tinynet_b', 'tinynet_c',
                        'tinynet_d', 'tinynet_e']:
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
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model_retrieval_functions = {
        'fastvit_t8': get_fastvit_model,
        'fastvit_t12': get_fastvit_model,
        'fastvit_s12': get_fastvit_model,
        'fastvit_sa12': get_fastvit_model,
        'fastvit_sa24': get_fastvit_model,
        'fastvit_sa36': get_fastvit_model,
        'fastvit_ma36': get_fastvit_model,
        'focalnet_tiny_srf': get_focalnet_model,
        'focalnet_small_srf': get_focalnet_model,
        'focalnet_base_srf': get_focalnet_model,
        'focalnet_tiny_lrf': get_focalnet_model,
        'focalnet_small_lrf': get_focalnet_model,
        'focalnet_base_lrf': get_focalnet_model,
        'focalnet_large_fl3': get_focalnet_model,
        'focalnet_large_fl4': get_focalnet_model,
        'focalnet_xlarge_fl3': get_focalnet_model,
        'focalnet_xlarge_fl4': get_focalnet_model,
        'focalnet_huge_fl3': get_focalnet_model,
        'focalnet_huge_fl4': get_focalnet_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError("Unknown model_type provided")

    return model_retrieval_functions[model_type](model_type, num_classes)


def get_family_model_g(model_type, num_classes):
    """
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model_retrieval_functions = {
        'gcvit_xxtiny': get_gcvit_model,
        'gcvit_xtiny': get_gcvit_model,
        'gcvit_tiny': get_gcvit_model,
        'gcvit_small': get_gcvit_model,
        'gcvit_base': get_gcvit_model,
        'ghostnet_050': get_ghostnet_model,
        'ghostnet_100': get_ghostnet_model,
        'gostnet_130': get_ghostnet_model,  # Typo corrected from 'ghostnet_130' to 'gostnet_130'
        'ghostnetv2_100': get_ghostnet_model,
        'ghostnetv2_130': get_ghostnet_model,
        'ghostnetv2_160': get_ghostnet_model,
        'GoogLeNet': get_googlenet_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError("Unknown model_type provided")

    return model_retrieval_functions[model_type](model_type, num_classes)


def get_family_model_h(model_type, num_classes):
    """
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model_retrieval_functions = {
        'hardcorenas_a': get_hardcorenas_model,
        'hardcorenas_b': get_hardcorenas_model,
        'hardcorenas_c': get_hardcorenas_model,
        'hardcorenas_d': get_hardcorenas_model,
        'hardcorenas_e': get_hardcorenas_model,
        'hardcorenas_f': get_hardcorenas_model,
        'hgnet_tiny': get_hgnet_model,
        'hgnet_small': get_hgnet_model,
        'hgnet_base': get_hgnet_model,
        'hgnetv2_b0': get_hgnet_model,
        'hgnetv2_b1': get_hgnet_model,
        'hgnetv2_b2': get_hgnet_model,
        'hgnetv2_b3': get_hgnet_model,
        'hgnetv2_b4': get_hgnet_model,
        'hgnetv2_b5': get_hgnet_model,
        'hgnetv2_b6': get_hgnet_model,
        'hrnet_w18_small': get_hrnet_model,
        'hrnet_w18_small_v2': get_hrnet_model,
        'hrnet_w18': get_hrnet_model,
        'hrnet_w30': get_hrnet_model,
        'hrnet_w32': get_hrnet_model,
        'hrnet_w40': get_hrnet_model,
        'hrnet_w44': get_hrnet_model,
        'hrnet_w48': get_hrnet_model,
        'hrnet_w64': get_hrnet_model,
        'hrnet_w18_ssld': get_hrnet_model,
        'hrnet_w48_ssld': get_hrnet_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError("Unknown model_type provided")

    return model_retrieval_functions[model_type](model_type, num_classes)


def get_family_model_i(model_type, num_classes):
    """
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model_retrieval_functions = {
        'Inception_V3': get_inception_model,
        'inception_v4': get_inception_model,
        'inception_resnet_v2': get_inception_model,
        'inception_next_tiny': get_inception_next_model,
        'inception_next_small': get_inception_next_model,
        'inception_next_base': get_inception_next_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError("Unknown model_type provided")

    return model_retrieval_functions[model_type](model_type, num_classes)


def get_family_model_l(model_type, num_classes):
    """
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model_retrieval_functions = {
        'levit_128s': get_levit_model,
        'levit_128': get_levit_model,
        'levit_192': get_levit_model,
        'levit_256': get_levit_model,
        'levit_384': get_levit_model,
        'levit_384_s8': get_levit_model,
        'levit_512_s8': get_levit_model,
        'levit_512': get_levit_model,
        'levit_256d': get_levit_model,
        'levit_512d': get_levit_model,
        'levit_conv_128s': get_levit_model,
        'levit_conv_128': get_levit_model,
        'levit_conv_192': get_levit_model,
        'levit_conv_256': get_levit_model,
        'levit_conv_384': get_levit_model,
        'levit_conv_384_s8': get_levit_model,
        'levit_conv_512_s8': get_levit_model,
        'levit_conv_512': get_levit_model,
        'levit_conv_256d': get_levit_model,
        'levit_conv_512d': get_levit_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError("Unknown model_type provided")

    return model_retrieval_functions[model_type](model_type, num_classes)


def get_family_model_m(model_type, num_classes):
    """
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model = "Unknown"

    if model_type in ['MaxVit_T', 'coatnet_pico_rw_224', 'coatnet_nano_rw_224', 'coatnet_0_rw_224',
                      'coatnet_1_rw_224', 'coatnet_2_rw_224', 'coatnet_3_rw_224', 'coatnet_bn_0_rw_224',
                      'coatnet_rmlp_nano_rw_224', 'coatnet_rmlp_0_rw_224', 'coatnet_rmlp_1_rw_224',
                      'coatnet_rmlp_1_rw2_224', 'coatnet_rmlp_2_rw_224', 'coatnet_rmlp_2_rw_384',
                      'coatnet_rmlp_3_rw_224', 'coatnet_nano_cc_224', 'coatnext_nano_rw_224', 'coatnet_0_224',
                      'coatnet_1_224', 'coatnet_2_224', 'coatnet_3_224', 'coatnet_4_224', 'coatnet_5_224',
                      'maxvit_pico_rw_256', 'maxvit_nano_rw_256', 'maxvit_tiny_rw_224', 'maxvit_tiny_rw_256',
                      'maxvit_rmlp_pico_rw_256', 'maxvit_rmlp_nano_rw_256', 'maxvit_rmlp_tiny_rw_256',
                      'maxvit_rmlp_small_rw_224', 'maxvit_rmlp_small_rw_256', "maxvit_rmlp_base_rw_224",
                      'maxvit_rmlp_base_rw_384', 'maxvit_tiny_pm_256', 'maxxvit_rmlp_nano_rw_256',
                      'maxxvit_rmlp_tiny_rw_256', 'maxxvit_rmlp_small_rw_256', 'maxxvitv2_nano_rw_256',
                      'maxxvitv2_rmlp_base_rw_224', 'maxxvitv2_rmlp_base_rw_384', 'maxxvitv2_rmlp_large_rw_224',
                      'maxvit_tiny_tf_224', 'maxvit_tiny_tf_384', 'maxvit_tiny_tf_512', 'maxvit_small_tf_224',
                      'maxvit_small_tf_384', 'maxvit_small_tf_512', 'maxvit_base_tf_224', 'maxvit_base_tf_384',
                      'maxvit_base_tf_512', 'maxvit_large_tf_224', 'maxvit_large_tf_384', 'maxvit_large_tf_512',
                      'maxvit_xlarge_tf_224', 'maxvit_xlarge_tf_384', 'maxvit_xlarge_tf_512']:
        model = get_maxvit_model(model_type, num_classes)
    elif model_type in ['MNASNet0_5', 'MNASNet0_75', 'MNASNet1_0', 'MNASNet1_3']:
        model = get_mnasnet_model(model_type, num_classes)
    elif model_type in ['poolformer_s12', 'poolformer_s24', 'poolformer_s36', 'poolformer_m36', 'poolformer_m48',
                        'poolformerv2_s12', 'poolformerv2_s24', 'poolformerv2_s36', 'poolformerv2_m36',
                        'poolformerv2_m48', 'convformer_s18', 'convformer_s36', 'convformer_m36', 'convformer_b36',
                        'caformer_s18', 'caformer_s36', 'caformer_m36', 'caformer_b36']:
        model = get_metaformer_model(model_type, num_classes)
    elif model_type in ['mixer_s32_224', 'mixer_s16_224', 'mixer_b32_224', 'mixer_b16_224', 'mixer_l32_224',
                        'mixer_l16_224', 'gmixer_12_224', 'gmixer_24_224', 'resmlp_12_224', 'resmlp_24_224',
                        'resmlp_36_224', 'resmlp_big_24_224', 'gmlp_ti16_224', 'gmlp_s16_224', 'gmlp_b16_224']:
        model = get_mlp_mixer_model(model_type, num_classes)
    elif model_type in ['MobileNet_V2', 'MobileNet_V3_Small', 'MobileNet_V3_Large', 'mobilenetv3_large_075',
                        'mobilenetv3_large_100', 'mobilenetv3_small_050', 'mobilenetv3_small_075',
                        'mobilenetv3_small_100', 'mobilenetv3_rw', 'tf_mobilenetv3_large_075',
                        'tf_mobilenetv3_large_100', 'tf_mobilenetv3_large_minimal_100', 'tf_mobilenetv3_small_075',
                        'tf_mobilenetv3_small_100', 'tf_mobilenetv3_small_minimal_100', 'fbnetv3_b', 'fbnetv3_d',
                        'fbnetv3_g', 'lcnet_035', 'lcnet_050', 'lcnet_075', 'lcnet_100', 'lcnet_150']:
        model = get_mobilenet_model(model_type, num_classes)
    elif model_type in ['mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s', 'mobilevitv2_050', 'mobilevitv2_075',
                        'mobilevitv2_100', 'mobilevitv2_125', 'mobilevitv2_150', 'mobilevitv2_175', 'mobilevitv2_200']:
        model = get_mobilevit_model(model_type, num_classes)
    elif model_type in ['mvitv2_tiny', 'mvitv2_small', 'mvitv2_base', 'mvitv2_large', 'mvitv2_small_cls',
                        'mvitv2_base_cls', 'mvitv2_large_cls', 'mvitv2_huge_cls']:
        model = get_mvitv2_model(model_type, num_classes)
    return model


def get_family_model_n(model_type, num_classes):
    """
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model = "Unknown"

    if model_type in ['nasnetalarge']:
        model = get_nasnet_model(model_type, num_classes)
    elif model_type in ['nest_base', 'nest_small', 'nest_tiny', 'nest_base_jx', 'nest_small_jx', 'nest_tiny_jx']:
        model = get_nest_model(model_type, num_classes)
    elif model_type in ['nextvit_small', 'nextvit_base', 'nextvit_large']:
        model = get_nextvit_model(model_type, num_classes)
    elif model_type in ['dm_nfnet_f0', 'dm_nfnet_f1', 'dm_nfnet_f2', 'dm_nfnet_f3', 'dm_nfnet_f4', 'dm_nfnet_f5',
                        'dm_nfnet_f6', 'nfnet_f0', 'nfnet_f1', 'nfnet_f2', 'nfnet_f3', 'nfnet_f4', 'nfnet_f5',
                        'nfnet_f6', 'nfnet_f7', 'nfnet_l0', 'eca_nfnet_l0', 'eca_nfnet_l1', 'eca_nfnet_l2',
                        'eca_nfnet_l3', 'nf_regnet_b0', 'nf_regnet_b1', 'nf_regnet_b2', 'nf_regnet_b3', 'nf_regnet_b4',
                        'nf_regnet_b5', 'nf_resnet26', 'nf_resnet50', 'nf_resnet101', 'nf_seresnet26', 'nf_seresnet50',
                        'nf_seresnet101', 'nf_ecaresnet26', 'nf_ecaresnet50', 'nf_ecaresnet101']:
        model = get_nfnet_model(model_type, num_classes)
    return model


def get_family_model_p(model_type, num_classes):
    """
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model = "Unknown"

    if model_type in ['pit_b_224', 'pit_s_224', 'pit_xs_224', 'pit_ti_224', 'pit_b_distilled_224',
                      'pit_s_distilled_224', 'pit_xs_distilled_224', 'pit_ti_distilled_224']:
        model = get_pit_model(model_type, num_classes)
    elif model_type == 'pnasnet5large':
        model = get_pnasnet_model(model_type, num_classes)
    elif model_type in ['pvt_v2_b0', 'pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5', 'pvt_v2_b2_li']:
        model = get_pvt_v2_model(model_type, num_classes)
    return model


def get_family_model_r(model_type, num_classes):
    """
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model = "Unknown"

    if model_type in ['RegNet_X_400MF', 'RegNet_X_800MF', 'RegNet_X_1_6GF', 'RegNet_X_3_2GF', 'RegNet_X_16GF',
                      'RegNet_Y_400MF', 'RegNet_Y_800MF', 'RegNet_Y_1_6GF', 'RegNet_Y_3_2GF', 'RegNet_Y_16GF',
                      'regnetx_002', 'regnetx_004', 'regnetx_004_tv', 'regnetx_006', 'regnetx_008', 'regnetx_016',
                      'regnetx_032', 'regnetx_040', 'regnetx_064', 'regnetx_080', 'regnetx_120', 'regnetx_160',
                      'regnetx_320', 'regnety_002', 'regnety_004', 'regnety_006', 'regnety_008', 'regnety_008_tv',
                      'regnety_016', 'regnety_032', 'regnety_040', 'regnety_064', 'regnety_080', 'regnety_080_tv',
                      'regnety_120', 'regnety_160', 'regnety_320', 'regnety_640', 'regnety_1280', 'regnety_2560',
                      'regnety_040_sgn', 'regnetv_040', 'regnetv_064', 'regnetz_005', 'regnetz_040', 'regnetz_040_h']:
        model = get_regnet_model(model_type, num_classes)
    elif model_type in ['repghostnet_050', 'repghostnet_058', 'repghostnet_080', 'repghostnet_100', 'repghostnet_111',
                        'repghostnet_130', 'repghostnet_150', 'repghostnet_200']:
        model = get_repghost_model(model_type, num_classes)
    elif model_type in ['repvit_m1', 'repvit_m2', 'repvit_m3', 'repvit_m0_9', 'repvit_m1_0', 'repvit_m1_1',
                        'repvit_m1_5', 'repvit_m2_3']:
        model = get_repvit_model(model_type, num_classes)
    elif model_type in ['res2net50_26w_4s', 'res2net101_26w_4s', 'res2net50_26w_6s', 'res2net50_26w_8s',
                        'res2net50_48w_2s', 'res2net50_14w_8s', 'res2next50', 'res2net50d', 'res2net101d']:
        model = get_res2net_model(model_type, num_classes)
    elif model_type in ['resnest14d', 'resnest26d', 'resnest50d', 'resnest101e', 'resnest200e', 'resnest269e',
                        'resnest50d_4s2x40d', 'resnest50d_1s4x24d']:
        model = get_resnest_model(model_type, num_classes)
    elif model_type in ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'resnet10t', 'resnet14t',
                        'resnet18', 'resnet18d', 'resnet34', 'resnet34d', 'resnet26', 'resnet26t', 'resnet26d',
                        'resnet50', 'resnet50c', 'resnet50d', 'resnet50s', 'resnet50t', 'resnet101', 'resnet101c',
                        'resnet101d', 'resnet101s', 'resnet152', 'resnet152c', 'resnet152d', 'resnet152s', 'resnet200',
                        'resnet200d', 'wide_resnet50_2', 'wide_resnet101_2', 'resnet50_gn', 'resnext50_32x4d',
                        'resnext50d_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_32x16d',
                        'resnext101_32x32d', 'resnext101_64x4d', 'ecaresnet26t', 'ecaresnet50d', 'ecaresnet50d_pruned',
                        'ecaresnet50t', 'ecaresnetlight', 'ecaresnet101d', 'ecaresnet101d_pruned', 'ecaresnet200d',
                        'ecaresnet269d', 'ecaresnext26t_32x4d', 'ecaresnext50t_32x4d', 'seresnet18', 'seresnet34',
                        'seresnet50', 'seresnet50t', 'seresnet101', 'seresnet152', 'seresnet152d', 'seresnet200d',
                        'seresnet269d', 'seresnext26d_32x4d', 'seresnext26t_32x4d', 'seresnext50_32x4d',
                        'seresnext101_32x8d', 'seresnext101d_32x8d', 'seresnext101_64x4d', 'senet154', 'resnetblur18',
                        'resnetblur50', 'resnetblur50d', 'resnetblur101d', 'resnetaa34d', 'resnetaa50', 'resnetaa50d',
                        'resnetaa101d', 'seresnetaa50d', 'seresnextaa101d_32x8d', 'seresnextaa201d_32x8d', 'resnetrs50',
                        'resnetrs101', 'resnetrs152', 'resnetrs200', 'resnetrs270', 'resnetrs350', 'resnetrs420']:
        model = get_resnet_model(model_type, num_classes)
    elif model_type in ['resnetv2_50x1_bit', 'resnetv2_50x3_bit', 'resnetv2_101x1_bit', 'resnetv2_101x3_bit',
                        'resnetv2_152x2_bit', 'resnetv2_152x4_bit', 'resnetv2_50', 'resnetv2_50d', 'resnetv2_50t',
                        'resnetv2_101', 'resnetv2_101d', 'resnetv2_152', 'resnetv2_152d', 'resnetv2_50d_gn',
                        'resnetv2_50d_evos', 'resnetv2_50d_frn']:
        model = get_resnetv2_model(model_type, num_classes)
    elif model_type in ['ResNeXt50_32X4D', 'ResNeXt101_32X8D', 'ResNeXt101_64X4D']:
        model = get_resnext_model(model_type, num_classes)
    elif model_type in ['rexnet_100', 'rexnet_130', 'rexnet_150', 'rexnet_200', 'rexnet_300', 'rexnetr_100',
                        'rexnetr_130', 'rexnetr_150', 'rexnetr_200', 'rexnetr_300']:
        model = get_rexnet_model(model_type, num_classes)

    return model


def get_family_model_s(model_type, num_classes):
    """
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model = "Unknown"

    if model_type in ['selecsls42', 'selecsls42b', 'selecsls60', 'selecsls60b', 'selecsls84']:
        model = get_selecsls_model(model_type, num_classes)
    elif model_type in ['legacy_seresnet18', 'legacy_seresnet34', 'legacy_seresnet50', 'legacy_seresnet101',
                        'legacy_seresnet152', 'legacy_senet154', 'legacy_seresnext26_32x4d',
                        'legacy_seresnext50_32x4d', 'legacy_seresnext101_32x4d']:
        model = get_senet_model(model_type, num_classes)
    elif model_type in ['sequencer2d_s', 'sequencer2d_m', 'sequencer2d_l']:
        model = get_sequencer_model(model_type, num_classes)
    elif model_type in ['ShuffleNet_V2_X0_5', 'ShuffleNet_V2_X1_0', 'ShuffleNet_V2_X1_5', 'ShuffleNet_V2_X2_0']:
        model = get_shufflenet_model(model_type, num_classes)
    elif model_type in ['skresnet18', 'skresnet34', 'skresnet50', 'skresnet50d', 'skresnext50_32x4d']:
        model = get_sknet_model(model_type, num_classes)
    elif model_type in ["SqueezeNet1_0", 'SqueezeNet1_1']:
        model = get_squeezenet_model(model_type, num_classes)
    elif model_type in ['Swin_T', 'Swin_S', 'Swin_B', 'Swin_V2_T', 'Swin_V2_S', 'Swin_V2_B',
                        'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224', 'swin_base_patch4_window7_224',
                        'swin_base_patch4_window12_384', 'swin_large_patch4_window7_224', 'swin_s3_tiny_224',
                        'swin_large_patch4_window12_384', 'swin_s3_small_224', 'swin_s3_base_224']:
        model = get_swin_transformer_model(model_type, num_classes)
    elif model_type in ["swinv2_tiny_window16_256", "swinv2_tiny_window8_256", "swinv2_small_window16_256",
                        "swinv2_small_window8_256", "swinv2_base_window16_256", "swinv2_base_window8_256",
                        "swinv2_base_window12_192", "swinv2_base_window12to16_192to256",
                        "swinv2_base_window12to24_192to384", "swinv2_large_window12_192",
                        "swinv2_large_window12to16_192to256", "swinv2_large_window12to24_192to384"]:
        model = get_swin_transformer_v2_model(model_type, num_classes)
    elif model_type in ["swinv2_cr_tiny_384", "swinv2_cr_tiny_224", "swinv2_cr_tiny_ns_224", "swinv2_cr_small_384",
                        "swinv2_cr_small_224", "swinv2_cr_small_ns_224", "swinv2_cr_small_ns_256", "swinv2_cr_base_384",
                        "swinv2_cr_base_224", "swinv2_cr_base_ns_224", "swinv2_cr_large_384", "swinv2_cr_large_224",
                        "swinv2_cr_huge_384", "swinv2_cr_huge_224", "swinv2_cr_giant_384", "swinv2_cr_giant_224"]:
        model = get_swin_transformer_v2_cr_model(model_type, num_classes)
    return model


def get_family_model_t(model_type, num_classes):
    """
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model = "Unknown"

    if model_type in ['tiny_vit_5m_224', 'tiny_vit_11m_224', 'tiny_vit_21m_224', 'tiny_vit_21m_384',
                      'tiny_vit_21m_512']:
        model = get_tiny_vit_model(model_type, num_classes)
    if model_type in ['tnt_s_patch16_224', 'tnt_b_patch16_224']:
        model = get_tnt_model(model_type, num_classes)
    elif model_type in ['tresnet_m', 'tresnet_l', 'tresnet_xl', 'tresnet_v2_l']:
        model = get_tresnet_model(model_type, num_classes)
    elif model_type in ['twins_pcpvt_small', 'twins_pcpvt_base', 'twins_pcpvt_large', 'twins_svt_small',
                        'twins_svt_base', 'twins_svt_large']:
        model = get_twins_model(model_type, num_classes)
    return model


def get_family_model_v(model_type, num_classes):
    """
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model = "Unknown"

    if model_type in ['VGG11', 'VGG11_BN', 'VGG13', 'VGG13_BN', 'VGG16', 'VGG16_BN', 'VGG19', 'VGG19_BN', 'vgg11',
                      'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']:
        model = get_vgg_model(model_type, num_classes)
    if model_type in ['visformer_tiny', 'visformer_small']:
        model = get_visformer_model(model_type, num_classes)
    elif model_type in ['ViT_B_16', 'ViT_B_32', 'ViT_L_16', 'ViT_L_32', 'ViT_H_14', "vit_tiny_patch16_224",
                        "vit_tiny_patch16_384", "vit_small_patch32_224", "vit_small_patch32_384",
                        "vit_small_patch16_224", "vit_small_patch16_384", "vit_small_patch8_224",
                        "vit_base_patch32_224", "vit_base_patch32_384", "vit_base_patch16_224",
                        "vit_base_patch16_384", "vit_base_patch8_224", "vit_large_patch32_224",
                        "vit_large_patch32_384", "vit_large_patch16_224", "vit_large_patch16_384",
                        "vit_large_patch14_224", "vit_large_patch14_224", "vit_giant_patch14_224",
                        "vit_gigantic_patch14_224", "vit_base_patch16_224_miil", "vit_medium_patch16_gap_240",
                        "vit_medium_patch16_gap_256", "vit_medium_patch16_gap_384", "vit_base_patch16_gap_224",
                        "vit_huge_patch14_gap_224", "vit_huge_patch16_gap_448", "vit_giant_patch16_gap_224",
                        "vit_xsmall_patch16_clip_224", "vit_medium_patch32_clip_224", "vit_medium_patch16_clip_224",
                        "vit_betwixt_patch32_clip_224", "vit_base_patch32_clip_224", "vit_base_patch32_clip_256",
                        "vit_base_patch32_clip_384", "vit_base_patch32_clip_448", "vit_base_patch16_clip_224",
                        "vit_base_patch16_clip_384", "vit_large_patch14_clip_224", "vit_large_patch14_clip_336",
                        "vit_huge_patch14_clip_224", "vit_huge_patch14_clip_336", "vit_huge_patch14_clip_378",
                        "vit_giant_patch14_clip_224", "vit_gigantic_patch14_clip_224",
                        "vit_base_patch32_clip_quickgelu_224", "vit_base_patch16_clip_quickgelu_224",
                        "vit_large_patch14_clip_quickgelu_224", "vit_large_patch14_clip_quickgelu_336",
                        "vit_huge_patch14_clip_quickgelu_224", "vit_huge_patch14_clip_quickgelu_378",
                        "vit_base_patch32_plus_256", "vit_base_patch16_plus_240", "vit_base_patch16_rpn_224",
                        "vit_small_patch16_36x1_224", "vit_small_patch16_18x2_224", "vit_base_patch16_18x2_224",
                        "eva_large_patch14_196", "eva_large_patch14_336", "flexivit_small", "flexivit_base",
                        "flexivit_large", "vit_base_patch16_xp_224", "vit_large_patch14_xp_224",
                        "vit_huge_patch14_xp_224", "vit_small_patch14_dinov2", "vit_base_patch14_dinov2",
                        "vit_large_patch14_dinov2", "vit_giant_patch14_dinov2", "vit_small_patch14_reg4_dinov2",
                        "vit_base_patch14_reg4_dinov2", "vit_large_patch14_reg4_dinov2",
                        "vit_giant_patch14_reg4_dinov2", "vit_base_patch16_siglip_224", "vit_base_patch16_siglip_256",
                        "vit_base_patch16_siglip_384", "vit_base_patch16_siglip_512", "vit_large_patch16_siglip_256",
                        "vit_large_patch16_siglip_384", "vit_so400m_patch14_siglip_224",
                        "vit_so400m_patch14_siglip_384", "vit_medium_patch16_reg4_gap_256",
                        "vit_base_patch16_reg4_gap_256", "vit_so150m_patch16_reg4_map_256",
                        "vit_so150m_patch16_reg4_gap_256"]:
        model = get_vision_transformer_model(model_type, num_classes)
    elif model_type in ['vit_tiny_r_s16_p8_224', 'vit_tiny_r_s16_p8_384', 'vit_small_r26_s32_224',
                        'vit_small_r26_s32_384', 'vit_base_r26_s32_224', 'vit_base_r50_s16_224',
                        'vit_base_r50_s16_384', 'vit_large_r50_s32_224', 'vit_large_r50_s32_384',
                        'vit_small_resnet26d_224', 'vit_small_resnet50d_s16_224', 'vit_base_resnet26d_224',
                        'vit_base_resnet50d_224']:
        model = get_vision_transformer_hybrid_model(model_type, num_classes)
    elif model_type in ['vit_relpos_base_patch32_plus_rpn_256', 'vit_relpos_base_patch16_plus_240',
                        'vit_relpos_small_patch16_224', 'vit_relpos_medium_patch16_224', 'vit_relpos_base_patch16_224',
                        'vit_srelpos_small_patch16_224', 'vit_srelpos_medium_patch16_224',
                        'vit_relpos_medium_patch16_cls_224', 'vit_relpos_base_patch16_cls_224',
                        'vit_relpos_base_patch16_clsgap_224', 'vit_relpos_small_patch16_rpn_224',
                        'vit_relpos_medium_patch16_rpn_224', 'vit_relpos_base_patch16_rpn_224']:
        model = get_vision_transformer_relpos_model(model_type, num_classes)
    elif model_type in ['samvit_base_patch16', 'samvit_large_patch16', 'samvit_huge_patch16',
                        'samvit_base_patch16_224']:
        model = get_vision_transformer_sam_model(model_type, num_classes)
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
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model_mapping = {
        'Wide_ResNet50_2': get_wide_resnet_model,
        'Wide_ResNet101_2': get_wide_resnet_model
    }

    if model_type in model_mapping:
        return model_mapping[model_type](model_type, num_classes)
    else:
        raise ValueError("Unknown model_type: {}".format(model_type))


def get_family_model_x(model_type, num_classes):
    """
    Retrieves a model from various families based on the provided model_type.

    Args:
        model_type (str): Type of the model to retrieve.
        num_classes (int): Number of output classes for the model.

    Returns:
        str: The retrieved model.

    Raises:
        ValueError: If an unknown model_type is provided.
    """
    model_mapping = {
        'legacy_xception': get_xception_model,
        'xception41': get_xception_model,
        'xception65': get_xception_model,
        'xception71': get_xception_model,
        'xception41p': get_xception_model,
        'xception65p': get_xception_model,
        'xcit_nano_12_p16_224': get_xcit_model,
        'xcit_nano_12_p16_384': get_xcit_model,
        'xcit_tiny_12_p16_224': get_xcit_model,
        'xcit_tiny_12_p16_384': get_xcit_model,
        'xcit_small_12_p16_224': get_xcit_model,
        'xcit_small_12_p16_384': get_xcit_model,
        'xcit_tiny_24_p16_224': get_xcit_model,
        'xcit_tiny_24_p16_384': get_xcit_model,
        'xcit_small_24_p16_224': get_xcit_model,
        'xcit_small_24_p16_384': get_xcit_model,
        'xcit_medium_24_p16_224': get_xcit_model,
        'xcit_medium_24_p16_384': get_xcit_model,
        'xcit_large_24_p16_224': get_xcit_model,
        'xcit_large_24_p16_384': get_xcit_model,
        'xcit_nano_12_p8_224': get_xcit_model,
        'xcit_nano_12_p8_384': get_xcit_model,
        'xcit_tiny_12_p8_224': get_xcit_model,
        'xcit_tiny_12_p8_384': get_xcit_model,
        'xcit_small_12_p8_224': get_xcit_model,
        'xcit_small_12_p8_384': get_xcit_model,
        'xcit_tiny_24_p8_224': get_xcit_model,
        'xcit_tiny_24_p8_384': get_xcit_model,
        'xcit_small_24_p8_224': get_xcit_model,
        'xcit_small_24_p8_384': get_xcit_model,
        'xcit_medium_24_p8_224': get_xcit_model,
        'xcit_medium_24_p8_384': get_xcit_model,
        'xcit_large_24_p8_224': get_xcit_model,
        'xcit_large_24_p8_384': get_xcit_model,
    }

    # Check if model_type is in the mapping, otherwise raise an error
    if model_type in model_mapping:
        return model_mapping[model_type](model_type, num_classes)
    else:
        raise ValueError("Unknown model_type: {}".format(model_type))


def get_family_model(model_type, num_classes):
    """
    Retrieves a model based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should start with a letter indicating the model family.
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model based on the provided model type and number of classes.
    """
    # Define a dictionary mapping first letters to functions
    model_functions = {
        'a': get_family_model_a,
        'b': get_family_model_b,
        'c': get_family_model_c,
        'e': get_family_model_e,
        'f': get_family_model_f,
        'g': get_family_model_g,
        'h': get_family_model_h,
        'i': get_family_model_i,
        'l': get_family_model_l,
        'm': get_family_model_m,
        'n': get_family_model_n,
        'p': get_family_model_p,
        'r': get_family_model_r,
        's': get_family_model_s,
        't': get_family_model_t,
        'v': get_family_model_v,
        'w': get_family_model_w,
        'x': get_family_model_x,
    }

    # Convert the first letter of the model_type to lowercase
    first_letter = model_type[0].lower()

    # Default value if no matching case is found
    model = "Error"

    # Retrieve the corresponding function and call it
    if first_letter in model_functions:
        model = model_functions[first_letter](model_type, num_classes)

    return model
