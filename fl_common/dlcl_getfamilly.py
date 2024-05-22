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
from fl_common.models.cvt import get_cvt_model
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
from fl_common.models.hiera import get_hiera_model
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
from fl_common.models.vision_transformer_hybrid import (
    get_vision_transformer_hybrid_model,
)
from fl_common.models.vision_transformer_relpos import (
    get_vision_transformer_relpos_model,
)
from fl_common.models.vision_transformer_sam import get_vision_transformer_sam_model
from fl_common.models.vitamin import get_vitamin_model
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
    # Dictionary mapping model_type to retrieval functions
    model_retrieval_functions = {"AlexNet": get_alexnet_model}

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

    return model_retrieval_functions[model_type](model_type, num_classes)


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
    # Dictionary mapping model_type to retrieval functions
    model_retrieval_functions = {
        "beit_base_patch16_224": get_beit_model,
        "beit_base_patch16_384": get_beit_model,
        "beit_large_patch16_224": get_beit_model,
        "beit_large_patch16_384": get_beit_model,
        "beit_large_patch16_512": get_beit_model,
        "beitv2_base_patch16_224": get_beit_model,
        "beitv2_large_patch16_224": get_beit_model,
        "botnet26t_256": get_byoanet_model,
        "sebotnet33ts_256": get_byoanet_model,
        "botnet50ts_256": get_byoanet_model,
        "eca_botnext26ts_256": get_byoanet_model,
        "halonet_h1": get_byoanet_model,
        "halonet26t": get_byoanet_model,
        "sehalonet33ts": get_byoanet_model,
        "halonet50ts": get_byoanet_model,
        "eca_halonext26ts": get_byoanet_model,
        "lambda_resnet26t": get_byoanet_model,
        "lambda_resnet50ts": get_byoanet_model,
        "lambda_resnet26rpt_256": get_byoanet_model,
        "haloregnetz_b": get_byoanet_model,
        "lamhalobotnet50ts_256": get_byoanet_model,
        "halo2botnet50ts_256": get_byoanet_model,
        "gernet_l": get_byobnet_model,
        "gernet_m": get_byobnet_model,
        "gernet_s": get_byobnet_model,
        "repvgg_a0": get_byobnet_model,
        "repvgg_a1": get_byobnet_model,
        "repvgg_a2": get_byobnet_model,
        "repvgg_b0": get_byobnet_model,
        "repvgg_b1": get_byobnet_model,
        "repvgg_b1g4": get_byobnet_model,
        "repvgg_b2": get_byobnet_model,
        "repvgg_b2g4": get_byobnet_model,
        "repvgg_b3": get_byobnet_model,
        "repvgg_b3g4": get_byobnet_model,
        "repvgg_d2se": get_byobnet_model,
        "resnet51q": get_byobnet_model,
        "resnet61q": get_byobnet_model,
        "resnext26ts": get_byobnet_model,
        "gcresnext26ts": get_byobnet_model,
        "seresnext26ts": get_byobnet_model,
        "eca_resnext26ts": get_byobnet_model,
        "bat_resnext26ts": get_byobnet_model,
        "resnet32ts": get_byobnet_model,
        "resnet33ts": get_byobnet_model,
        "gcresnet33ts": get_byobnet_model,
        "seresnet33ts": get_byobnet_model,
        "eca_resnet33ts": get_byobnet_model,
        "gcresnet50t": get_byobnet_model,
        "gcresnext50ts": get_byobnet_model,
        "regnetz_b16": get_byobnet_model,
        "regnetz_c16": get_byobnet_model,
        "regnetz_d32": get_byobnet_model,
        "regnetz_d8": get_byobnet_model,
        "regnetz_e8": get_byobnet_model,
        "regnetz_b16_evos": get_byobnet_model,
        "regnetz_c16_evos": get_byobnet_model,
        "regnetz_d8_evos": get_byobnet_model,
        "mobileone_s0": get_byobnet_model,
        "mobileone_s1": get_byobnet_model,
        "mobileone_s2": get_byobnet_model,
        "mobileone_s3": get_byobnet_model,
        "mobileone_s4": get_byobnet_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

    return model_retrieval_functions[model_type](model_type, num_classes)


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
        "cait_xxs24_224": get_cait_model,
        "cait_xxs24_384": get_cait_model,
        "cait_xxs36_224": get_cait_model,
        "cait_xxs36_384": get_cait_model,
        "cait_xs24_384": get_cait_model,
        "cait_s24_224": get_cait_model,
        "cait_s24_384": get_cait_model,
        "cait_s36_224": get_cait_model,
        "cait_m36_384": get_cait_model,
        "cait_m48_448": get_cait_model,
        "coat_tiny": get_coat_model,
        "coat_mini": get_coat_model,
        "coat_small": get_coat_model,
        "coat_lite_tiny": get_coat_model,
        "coat_lite_mini": get_coat_model,
        "coat_lite_small": get_coat_model,
        "coat_lite_medium": get_coat_model,
        "coat_lite_medium_384": get_coat_model,
        "convmixer_1536_20": get_convmixer_model,
        "convmixer_768_32": get_convmixer_model,
        "convmixer_1024_20_ks9_p14": get_convmixer_model,
        "convit_tiny": get_convit_model,
        "convit_small": get_convit_model,
        "convit_base": get_convit_model,
        "ConvNeXt_Tiny": get_convnext_model,
        "ConvNeXt_Small": get_convnext_model,
        "ConvNeXt_Base": get_convnext_model,
        "ConvNeXt_Large": get_convnext_model,
        "convnext_atto": get_convnext_model,
        "convnext_atto_ols": get_convnext_model,
        "convnext_femto": get_convnext_model,
        "convnext_femto_ols": get_convnext_model,
        "convnext_pico": get_convnext_model,
        "convnext_pico_ols": get_convnext_model,
        "convnext_nano": get_convnext_model,
        "convnext_nano_ols": get_convnext_model,
        "convnext_tiny_hnf": get_convnext_model,
        "convnext_tiny": get_convnext_model,
        "convnext_small": get_convnext_model,
        "convnext_base": get_convnext_model,
        "convnext_large": get_convnext_model,
        "convnext_large_mlp": get_convnext_model,
        "convnext_xlarge": get_convnext_model,
        "convnext_xxlarge": get_convnext_model,
        "convnextv2_atto": get_convnext_model,
        "convnextv2_femto": get_convnext_model,
        "convnextv2_pico": get_convnext_model,
        "convnextv2_nano": get_convnext_model,
        "convnextv2_tiny": get_convnext_model,
        "convnextv2_small": get_convnext_model,
        "convnextv2_base": get_convnext_model,
        "convnextv2_large": get_convnext_model,
        "convnextv2_huge": get_convnext_model,
        "crossvit_tiny_240": get_crossvit_model,
        "rossvit_small_240": get_crossvit_model,
        "crossvit_base_240": get_crossvit_model,
        "crossvit_9_240": get_crossvit_model,
        "crossvit_15_240": get_crossvit_model,
        "crossvit_18_240": get_crossvit_model,
        "crossvit_9_dagger_240": get_crossvit_model,
        "rossvit_15_dagger_240": get_crossvit_model,
        "crossvit_15_dagger_408": get_crossvit_model,
        "crossvit_18_dagger_240": get_crossvit_model,
        "crossvit_18_dagger_408": get_crossvit_model,
        "cspresnet50": get_cspnet_model,
        "cspresnet50d": get_cspnet_model,
        "cspresnet50w": get_cspnet_model,
        "cspresnext50": get_cspnet_model,
        "cspdarknet53": get_cspnet_model,
        "darknet17": get_cspnet_model,
        "darknet21": get_cspnet_model,
        "sedarknet21": get_cspnet_model,
        "darknet53": get_cspnet_model,
        "darknetaa53": get_cspnet_model,
        "cs3darknet_s": get_cspnet_model,
        "cs3darknet_m": get_cspnet_model,
        "cs3darknet_l": get_cspnet_model,
        "cs3darknet_x": get_cspnet_model,
        "cs3darknet_focus_s": get_cspnet_model,
        "cs3darknet_focus_m": get_cspnet_model,
        "cs3darknet_focus_l": get_cspnet_model,
        "cs3darknet_focus_x": get_cspnet_model,
        "cs3sedarknet_l": get_cspnet_model,
        "cs3sedarknet_x": get_cspnet_model,
        "cs3sedarknet_xdw": get_cspnet_model,
        "cs3edgenet_x": get_cspnet_model,
        "cs3se_edgenet_x": get_cspnet_model,
        "cvt_13": get_cvt_model,
        "cvt_21": get_cvt_model,
        "cvt_w24": get_cvt_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

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
        "davit_tiny": get_davit_model,
        "davit_small": get_davit_model,
        "davit_base": get_davit_model,
        "davit_large": get_davit_model,
        "davit_huge": get_davit_model,
        "davit_giant": get_davit_model,
        "deit_tiny_patch16_224": get_deit_model,
        "deit_small_patch16_224": get_deit_model,
        "deit_base_patch16_224": get_deit_model,
        "deit_base_patch16_384": get_deit_model,
        "deit_tiny_distilled_patch16_224": get_deit_model,
        "deit_small_distilled_patch16_224": get_deit_model,
        "deit_base_distilled_patch16_224": get_deit_model,
        "deit_base_distilled_patch16_384": get_deit_model,
        "deit3_small_patch16_224": get_deit_model,
        "deit3_small_patch16_384": get_deit_model,
        "deit3_medium_patch16_224": get_deit_model,
        "deit3_base_patch16_224": get_deit_model,
        "deit3_base_patch16_384": get_deit_model,
        "deit3_large_patch16_224": get_deit_model,
        "deit3_large_patch16_384": get_deit_model,
        "deit3_huge_patch14_224": get_deit_model,
        "DenseNet121": get_densenet_model,
        "DenseNet161": get_densenet_model,
        "DenseNet169": get_densenet_model,
        "DenseNet201": get_densenet_model,
        "densenet121": get_densenet_model,
        "densenetblur121d": get_densenet_model,
        "densenet169": get_densenet_model,
        "densenet201": get_densenet_model,
        "densenet161": get_densenet_model,
        "densenet264d": get_densenet_model,
        "dla60_res2net": get_dla_model,
        "dla60_res2next": get_dla_model,
        "dla34": get_dla_model,
        "dla46_c": get_dla_model,
        "dla46x_c": get_dla_model,
        "dla60x_c": get_dla_model,
        "dla60": get_dla_model,
        "dla60x": get_dla_model,
        "dla102": get_dla_model,
        "dla102x": get_dla_model,
        "dla102x2": get_dla_model,
        "dla169": get_dla_model,
        "dpn48b": get_dpn_model,
        "dpn68": get_dpn_model,
        "dpn68b": get_dpn_model,
        "dpn92": get_dpn_model,
        "dpn98": get_dpn_model,
        "dpn131": get_dpn_model,
        "dpn107": get_dpn_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

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
    # Dictionary mapping model_type to retrieval functions
    model_retrieval_functions = {
        "edgenext_xx_small": get_edgenet_model,
        "edgenext_x_small": get_edgenet_model,
        "edgenext_small": get_edgenet_model,
        "edgenext_base": get_edgenet_model,
        "edgenext_small_rw": get_edgenet_model,
        "efficientformer_l1": get_efficientformer_model,
        "efficientformer_l3": get_efficientformer_model,
        "efficientformer_l7": get_efficientformer_model,
        "efficientformerv2_s0": get_efficientformer_v2_model,
        "efficientformerv2_s1": get_efficientformer_v2_model,
        "efficientformerv2_s2": get_efficientformer_v2_model,
        "efficientformerv2_l": get_efficientformer_v2_model,
        "efficientvit_b0": get_efficientvit_mit_model,
        "efficientvit_b1": get_efficientvit_mit_model,
        "efficientvit_b2": get_efficientvit_mit_model,
        "efficientvit_b3": get_efficientvit_mit_model,
        "efficientvit_l1": get_efficientvit_mit_model,
        "efficientvit_l2": get_efficientvit_mit_model,
        "efficientvit_l3": get_efficientvit_mit_model,
        "efficientvit_m0": get_efficientvit_msra_model,
        "efficientvit_m1": get_efficientvit_msra_model,
        "efficientvit_m2": get_efficientvit_msra_model,
        "efficientvit_m3": get_efficientvit_msra_model,
        "efficientvit_m4": get_efficientvit_msra_model,
        "efficientvit_m5": get_efficientvit_msra_model,
        "EfficientNetB0": get_efficientnet_model,
        "EfficientNetB1": get_efficientnet_model,
        "EfficientNetB2": get_efficientnet_model,
        "EfficientNetB3": get_efficientnet_model,
        "EfficientNetB4": get_efficientnet_model,
        "EfficientNetB5": get_efficientnet_model,
        "EfficientNetB6": get_efficientnet_model,
        "EfficientNetB7": get_efficientnet_model,
        "EfficientNetV2S": get_efficientnet_model,
        "EfficientNetV2M": get_efficientnet_model,
        "EfficientNetV2L": get_efficientnet_model,
        "mnasnet_050": get_efficientnet_model,
        "mnasnet_075": get_efficientnet_model,
        "mnasnet_100": get_efficientnet_model,
        "mnasnet_140": get_efficientnet_model,
        "semnasnet_050": get_efficientnet_model,
        "semnasnet_075": get_efficientnet_model,
        "semnasnet_100": get_efficientnet_model,
        "semnasnet_140": get_efficientnet_model,
        "mnasnet_small": get_efficientnet_model,
        "mobilenetv2_035": get_efficientnet_model,
        "mobilenetv2_050": get_efficientnet_model,
        "mobilenetv2_075": get_efficientnet_model,
        "mobilenetv2_100": get_efficientnet_model,
        "mobilenetv2_140": get_efficientnet_model,
        "mobilenetv2_110d": get_efficientnet_model,
        "mobilenetv2_120d": get_efficientnet_model,
        "fbnetc_100": get_efficientnet_model,
        "spnasnet_100": get_efficientnet_model,
        "efficientnet_b0": get_efficientnet_model,
        "efficientnet_b1": get_efficientnet_model,
        "efficientnet_b2": get_efficientnet_model,
        "efficientnet_b3": get_efficientnet_model,
        "efficientnet_b4": get_efficientnet_model,
        "efficientnet_b5": get_efficientnet_model,
        "efficientnet_b6": get_efficientnet_model,
        "efficientnet_b7": get_efficientnet_model,
        "efficientnet_b8": get_efficientnet_model,
        "efficientnet_l2": get_efficientnet_model,
        "efficientnet_b0_gn": get_efficientnet_model,
        "efficientnet_b0_g8_gn": get_efficientnet_model,
        "efficientnet_b0_g16_evos": get_efficientnet_model,
        "efficientnet_b3_gn": get_efficientnet_model,
        "efficientnet_b3_g8_gn": get_efficientnet_model,
        "efficientnet_es": get_efficientnet_model,
        "efficientnet_es_pruned": get_efficientnet_model,
        "efficientnet_em": get_efficientnet_model,
        "efficientnet_el": get_efficientnet_model,
        "efficientnet_el_pruned": get_efficientnet_model,
        "efficientnet_cc_b0_4e": get_efficientnet_model,
        "efficientnet_cc_b0_8e": get_efficientnet_model,
        "efficientnet_cc_b1_8e": get_efficientnet_model,
        "efficientnet_lite0": get_efficientnet_model,
        "efficientnet_lite1": get_efficientnet_model,
        "efficientnet_lite2": get_efficientnet_model,
        "efficientnet_lite3": get_efficientnet_model,
        "efficientnet_lite4": get_efficientnet_model,
        "efficientnet_b1_pruned": get_efficientnet_model,
        "efficientnet_b2_pruned": get_efficientnet_model,
        "efficientnet_b3_pruned": get_efficientnet_model,
        "efficientnetv2_rw_t": get_efficientnet_model,
        "gc_efficientnetv2_rw_t": get_efficientnet_model,
        "efficientnetv2_rw_s": get_efficientnet_model,
        "efficientnetv2_rw_m": get_efficientnet_model,
        "efficientnetv2_s": get_efficientnet_model,
        "efficientnetv2_m": get_efficientnet_model,
        "efficientnetv2_l": get_efficientnet_model,
        "efficientnetv2_xl": get_efficientnet_model,
        "tf_efficientnet_b0": get_efficientnet_model,
        "tf_efficientnet_b1": get_efficientnet_model,
        "tf_efficientnet_b2": get_efficientnet_model,
        "tf_efficientnet_b3": get_efficientnet_model,
        "tf_efficientnet_b4": get_efficientnet_model,
        "tf_efficientnet_b5": get_efficientnet_model,
        "tf_efficientnet_b6": get_efficientnet_model,
        "tf_efficientnet_b7": get_efficientnet_model,
        "tf_efficientnet_b8": get_efficientnet_model,
        "tf_efficientnet_l2": get_efficientnet_model,
        "tf_efficientnet_es": get_efficientnet_model,
        "tf_efficientnet_em": get_efficientnet_model,
        "tf_efficientnet_el": get_efficientnet_model,
        "tf_efficientnet_cc_b0_4e": get_efficientnet_model,
        "tf_efficientnet_cc_b0_8e": get_efficientnet_model,
        "tf_efficientnet_cc_b1_8e": get_efficientnet_model,
        "tf_efficientnet_lite0": get_efficientnet_model,
        "tf_efficientnet_lite1": get_efficientnet_model,
        "tf_efficientnet_lite2": get_efficientnet_model,
        "tf_efficientnet_lite3": get_efficientnet_model,
        "tf_efficientnet_lite4": get_efficientnet_model,
        "tf_efficientnetv2_s": get_efficientnet_model,
        "tf_efficientnetv2_m": get_efficientnet_model,
        "tf_efficientnetv2_l": get_efficientnet_model,
        "tf_efficientnetv2_xl": get_efficientnet_model,
        "tf_efficientnetv2_b0": get_efficientnet_model,
        "tf_efficientnetv2_b1": get_efficientnet_model,
        "tf_efficientnetv2_b2": get_efficientnet_model,
        "tf_efficientnetv2_b3": get_efficientnet_model,
        "mixnet_s": get_efficientnet_model,
        "mixnet_m": get_efficientnet_model,
        "mixnet_l": get_efficientnet_model,
        "mixnet_xl": get_efficientnet_model,
        "mixnet_xxl": get_efficientnet_model,
        "tf_mixnet_s": get_efficientnet_model,
        "tf_mixnet_m": get_efficientnet_model,
        "tf_mixnet_l": get_efficientnet_model,
        "tinynet_a": get_efficientnet_model,
        "tinynet_b": get_efficientnet_model,
        "tinynet_c": get_efficientnet_model,
        "tinynet_d": get_efficientnet_model,
        "tinynet_e": get_efficientnet_model,
        "eva_giant_patch14_224": get_eva_model,
        "eva_giant_patch14_336": get_eva_model,
        "eva_giant_patch14_560": get_eva_model,
        "eva02_tiny_patch14_224": get_eva_model,
        "eva02_small_patch14_224": get_eva_model,
        "eva02_base_patch14_224": get_eva_model,
        "eva02_large_patch14_224": get_eva_model,
        "eva02_tiny_patch14_336": get_eva_model,
        "eva02_small_patch14_336": get_eva_model,
        "eva02_base_patch14_448": get_eva_model,
        "eva02_large_patch14_448": get_eva_model,
        "eva_giant_patch14_clip_224": get_eva_model,
        "eva02_base_patch16_clip_224": get_eva_model,
        "eva02_large_patch14_clip_224": get_eva_model,
        "eva02_large_patch14_clip_336": get_eva_model,
        "eva02_enormous_patch14_clip_224": get_eva_model,
        "vit_medium_patch16_rope_reg1_gap_256": get_eva_model,
        "vit_mediumd_patch16_rope_reg1_gap_256": get_eva_model,
        "vit_betwixt_patch16_rope_reg4_gap_256": get_eva_model,
        "vit_base_patch16_rope_reg1_gap_256": get_eva_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

    return model_retrieval_functions[model_type](model_type, num_classes)


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
        "fastvit_t8": get_fastvit_model,
        "fastvit_t12": get_fastvit_model,
        "fastvit_s12": get_fastvit_model,
        "fastvit_sa12": get_fastvit_model,
        "fastvit_sa24": get_fastvit_model,
        "fastvit_sa36": get_fastvit_model,
        "fastvit_ma36": get_fastvit_model,
        "focalnet_tiny_srf": get_focalnet_model,
        "focalnet_small_srf": get_focalnet_model,
        "focalnet_base_srf": get_focalnet_model,
        "focalnet_tiny_lrf": get_focalnet_model,
        "focalnet_small_lrf": get_focalnet_model,
        "focalnet_base_lrf": get_focalnet_model,
        "focalnet_large_fl3": get_focalnet_model,
        "focalnet_large_fl4": get_focalnet_model,
        "focalnet_xlarge_fl3": get_focalnet_model,
        "focalnet_xlarge_fl4": get_focalnet_model,
        "focalnet_huge_fl3": get_focalnet_model,
        "focalnet_huge_fl4": get_focalnet_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

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
        "gcvit_xxtiny": get_gcvit_model,
        "gcvit_xtiny": get_gcvit_model,
        "gcvit_tiny": get_gcvit_model,
        "gcvit_small": get_gcvit_model,
        "gcvit_base": get_gcvit_model,
        "ghostnet_050": get_ghostnet_model,
        "ghostnet_100": get_ghostnet_model,
        "ghostnet_130": get_ghostnet_model,
        "ghostnetv2_100": get_ghostnet_model,
        "ghostnetv2_130": get_ghostnet_model,
        "ghostnetv2_160": get_ghostnet_model,
        "GoogLeNet": get_googlenet_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

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
        "hardcorenas_a": get_hardcorenas_model,
        "hardcorenas_b": get_hardcorenas_model,
        "hardcorenas_c": get_hardcorenas_model,
        "hardcorenas_d": get_hardcorenas_model,
        "hardcorenas_e": get_hardcorenas_model,
        "hardcorenas_f": get_hardcorenas_model,
        "hgnet_tiny": get_hgnet_model,
        "hgnet_small": get_hgnet_model,
        "hgnet_base": get_hgnet_model,
        "hgnetv2_b0": get_hgnet_model,
        "hgnetv2_b1": get_hgnet_model,
        "hgnetv2_b2": get_hgnet_model,
        "hgnetv2_b3": get_hgnet_model,
        "hgnetv2_b4": get_hgnet_model,
        "hgnetv2_b5": get_hgnet_model,
        "hgnetv2_b6": get_hgnet_model,
        "hiera_tiny_224": get_hiera_model,
        "hiera_small_224": get_hiera_model,
        "hiera_base_224": get_hiera_model,
        "hiera_base_plus_224": get_hiera_model,
        "hiera_large_224": get_hiera_model,
        "hiera_huge_224": get_hiera_model,
        "hrnet_w18_small": get_hrnet_model,
        "hrnet_w18_small_v2": get_hrnet_model,
        "hrnet_w18": get_hrnet_model,
        "hrnet_w30": get_hrnet_model,
        "hrnet_w32": get_hrnet_model,
        "hrnet_w40": get_hrnet_model,
        "hrnet_w44": get_hrnet_model,
        "hrnet_w48": get_hrnet_model,
        "hrnet_w64": get_hrnet_model,
        "hrnet_w18_ssld": get_hrnet_model,
        "hrnet_w48_ssld": get_hrnet_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

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
        "Inception_V3": get_inception_model,
        "inception_v4": get_inception_model,
        "inception_resnet_v2": get_inception_model,
        "inception_next_tiny": get_inception_next_model,
        "inception_next_small": get_inception_next_model,
        "inception_next_base": get_inception_next_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

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
        "levit_128s": get_levit_model,
        "levit_128": get_levit_model,
        "levit_192": get_levit_model,
        "levit_256": get_levit_model,
        "levit_384": get_levit_model,
        "levit_384_s8": get_levit_model,
        "levit_512_s8": get_levit_model,
        "levit_512": get_levit_model,
        "levit_256d": get_levit_model,
        "levit_512d": get_levit_model,
        "levit_conv_128s": get_levit_model,
        "levit_conv_128": get_levit_model,
        "levit_conv_192": get_levit_model,
        "levit_conv_256": get_levit_model,
        "levit_conv_384": get_levit_model,
        "levit_conv_384_s8": get_levit_model,
        "levit_conv_512_s8": get_levit_model,
        "levit_conv_512": get_levit_model,
        "levit_conv_256d": get_levit_model,
        "levit_conv_512d": get_levit_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

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
    model_retrieval_functions = {
        "MaxVit_T": get_maxvit_model,
        "coatnet_pico_rw_224": get_maxvit_model,
        "coatnet_nano_rw_224": get_maxvit_model,
        "coatnet_0_rw_224": get_maxvit_model,
        "coatnet_1_rw_224": get_maxvit_model,
        "coatnet_2_rw_224": get_maxvit_model,
        "coatnet_3_rw_224": get_maxvit_model,
        "coatnet_bn_0_rw_224": get_maxvit_model,
        "coatnet_rmlp_nano_rw_224": get_maxvit_model,
        "coatnet_rmlp_0_rw_224": get_maxvit_model,
        "coatnet_rmlp_1_rw_224": get_maxvit_model,
        "coatnet_rmlp_1_rw2_224": get_maxvit_model,
        "coatnet_rmlp_2_rw_224": get_maxvit_model,
        "coatnet_rmlp_2_rw_384": get_maxvit_model,
        "coatnet_rmlp_3_rw_224": get_maxvit_model,
        "coatnet_nano_cc_224": get_maxvit_model,
        "coatnext_nano_rw_224": get_maxvit_model,
        "coatnet_0_224": get_maxvit_model,
        "coatnet_1_224": get_maxvit_model,
        "coatnet_2_224": get_maxvit_model,
        "coatnet_3_224": get_maxvit_model,
        "coatnet_4_224": get_maxvit_model,
        "coatnet_5_224": get_maxvit_model,
        "maxvit_pico_rw_256": get_maxvit_model,
        "maxvit_nano_rw_256": get_maxvit_model,
        "maxvit_tiny_rw_224": get_maxvit_model,
        "maxvit_tiny_rw_256": get_maxvit_model,
        "maxvit_rmlp_pico_rw_256": get_maxvit_model,
        "maxvit_rmlp_nano_rw_256": get_maxvit_model,
        "maxvit_rmlp_tiny_rw_256": get_maxvit_model,
        "maxvit_rmlp_small_rw_224": get_maxvit_model,
        "maxvit_rmlp_small_rw_256": get_maxvit_model,
        "maxvit_rmlp_base_rw_224": get_maxvit_model,
        "maxvit_rmlp_base_rw_384": get_maxvit_model,
        "maxvit_tiny_pm_256": get_maxvit_model,
        "maxxvit_rmlp_nano_rw_256": get_maxvit_model,
        "maxxvit_rmlp_tiny_rw_256": get_maxvit_model,
        "maxxvit_rmlp_small_rw_256": get_maxvit_model,
        "maxxvitv2_nano_rw_256": get_maxvit_model,
        "maxxvitv2_rmlp_base_rw_224": get_maxvit_model,
        "maxxvitv2_rmlp_base_rw_384": get_maxvit_model,
        "maxxvitv2_rmlp_large_rw_224": get_maxvit_model,
        "maxvit_tiny_tf_224": get_maxvit_model,
        "maxvit_tiny_tf_384": get_maxvit_model,
        "maxvit_tiny_tf_512": get_maxvit_model,
        "maxvit_small_tf_224": get_maxvit_model,
        "maxvit_small_tf_384": get_maxvit_model,
        "maxvit_small_tf_512": get_maxvit_model,
        "maxvit_base_tf_224": get_maxvit_model,
        "maxvit_base_tf_384": get_maxvit_model,
        "maxvit_base_tf_512": get_maxvit_model,
        "maxvit_large_tf_224": get_maxvit_model,
        "maxvit_large_tf_384": get_maxvit_model,
        "maxvit_large_tf_512": get_maxvit_model,
        "maxvit_xlarge_tf_224": get_maxvit_model,
        "maxvit_xlarge_tf_384": get_maxvit_model,
        "maxvit_xlarge_tf_512": get_maxvit_model,
        "MNASNet0_5": get_mnasnet_model,
        "MNASNet0_75": get_mnasnet_model,
        "MNASNet1_0": get_mnasnet_model,
        "MNASNet1_3": get_mnasnet_model,
        "poolformer_s12": get_metaformer_model,
        "poolformer_s24": get_metaformer_model,
        "poolformer_s36": get_metaformer_model,
        "poolformer_m36": get_metaformer_model,
        "poolformer_m48": get_metaformer_model,
        "poolformerv2_s12": get_metaformer_model,
        "poolformerv2_s24": get_metaformer_model,
        "poolformerv2_s36": get_metaformer_model,
        "poolformerv2_m36": get_metaformer_model,
        "poolformerv2_m48": get_metaformer_model,
        "convformer_s18": get_metaformer_model,
        "convformer_s36": get_metaformer_model,
        "convformer_m36": get_metaformer_model,
        "convformer_b36": get_metaformer_model,
        "caformer_s18": get_metaformer_model,
        "caformer_s36": get_metaformer_model,
        "caformer_m36": get_metaformer_model,
        "caformer_b36": get_metaformer_model,
        "mixer_s32_224": get_mlp_mixer_model,
        "mixer_s16_224": get_mlp_mixer_model,
        "mixer_b32_224": get_mlp_mixer_model,
        "mixer_b16_224": get_mlp_mixer_model,
        "mixer_l32_224": get_mlp_mixer_model,
        "mixer_l16_224": get_mlp_mixer_model,
        "gmixer_12_224": get_mlp_mixer_model,
        "gmixer_24_224": get_mlp_mixer_model,
        "resmlp_12_224": get_mlp_mixer_model,
        "resmlp_24_224": get_mlp_mixer_model,
        "resmlp_36_224": get_mlp_mixer_model,
        "resmlp_big_24_224": get_mlp_mixer_model,
        "gmlp_ti16_224": get_mlp_mixer_model,
        "gmlp_s16_224": get_mlp_mixer_model,
        "gmlp_b16_224": get_mlp_mixer_model,
        "MobileNet_V2": get_mobilenet_model,
        "MobileNet_V3_Small": get_mobilenet_model,
        "MobileNet_V3_Large": get_mobilenet_model,
        "mobilenetv3_large_075": get_mobilenet_model,
        "mobilenetv3_large_100": get_mobilenet_model,
        "mobilenetv3_small_050": get_mobilenet_model,
        "mobilenetv3_small_075": get_mobilenet_model,
        "mobilenetv3_small_100": get_mobilenet_model,
        "mobilenetv3_rw": get_mobilenet_model,
        "tf_mobilenetv3_large_075": get_mobilenet_model,
        "tf_mobilenetv3_large_100": get_mobilenet_model,
        "tf_mobilenetv3_large_minimal_100": get_mobilenet_model,
        "tf_mobilenetv3_small_075": get_mobilenet_model,
        "tf_mobilenetv3_small_100": get_mobilenet_model,
        "tf_mobilenetv3_small_minimal_100": get_mobilenet_model,
        "fbnetv3_b": get_mobilenet_model,
        "fbnetv3_d": get_mobilenet_model,
        "fbnetv3_g": get_mobilenet_model,
        "lcnet_035": get_mobilenet_model,
        "lcnet_050": get_mobilenet_model,
        "lcnet_075": get_mobilenet_model,
        "lcnet_100": get_mobilenet_model,
        "lcnet_150": get_mobilenet_model,
        "mobilevit_xxs": get_mobilevit_model,
        "mobilevit_xs": get_mobilevit_model,
        "mobilevit_s": get_mobilevit_model,
        "mobilevitv2_050": get_mobilevit_model,
        "mobilevitv2_075": get_mobilevit_model,
        "mobilevitv2_100": get_mobilevit_model,
        "mobilevitv2_125": get_mobilevit_model,
        "mobilevitv2_150": get_mobilevit_model,
        "mobilevitv2_175": get_mobilevit_model,
        "mobilevitv2_200": get_mobilevit_model,
        "mvitv2_tiny": get_mvitv2_model,
        "mvitv2_small": get_mvitv2_model,
        "mvitv2_base": get_mvitv2_model,
        "mvitv2_large": get_mvitv2_model,
        "mvitv2_small_cls": get_mvitv2_model,
        "mvitv2_base_cls": get_mvitv2_model,
        "mvitv2_large_cls": get_mvitv2_model,
        "mvitv2_huge_cls": get_mvitv2_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

    return model_retrieval_functions[model_type](model_type, num_classes)


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
    model_retrieval_functions = {
        "nasnetalarge": get_nasnet_model,
        "nest_base": get_nest_model,
        "nest_small": get_nest_model,
        "nest_tiny": get_nest_model,
        "nest_base_jx": get_nest_model,
        "nest_small_jx": get_nest_model,
        "nest_tiny_jx": get_nest_model,
        "nextvit_small": get_nextvit_model,
        "nextvit_base": get_nextvit_model,
        "nextvit_large": get_nextvit_model,
        "dm_nfnet_f0": get_nfnet_model,
        "dm_nfnet_f1": get_nfnet_model,
        "dm_nfnet_f2": get_nfnet_model,
        "dm_nfnet_f3": get_nfnet_model,
        "dm_nfnet_f4": get_nfnet_model,
        "dm_nfnet_f5": get_nfnet_model,
        "dm_nfnet_f6": get_nfnet_model,
        "nfnet_f0": get_nfnet_model,
        "nfnet_f1": get_nfnet_model,
        "nfnet_f2": get_nfnet_model,
        "nfnet_f3": get_nfnet_model,
        "nfnet_f4": get_nfnet_model,
        "nfnet_f5": get_nfnet_model,
        "nfnet_f6": get_nfnet_model,
        "nfnet_f7": get_nfnet_model,
        "nfnet_l0": get_nfnet_model,
        "eca_nfnet_l0": get_nfnet_model,
        "eca_nfnet_l1": get_nfnet_model,
        "eca_nfnet_l2": get_nfnet_model,
        "eca_nfnet_l3": get_nfnet_model,
        "nf_regnet_b0": get_nfnet_model,
        "nf_regnet_b1": get_nfnet_model,
        "nf_regnet_b2": get_nfnet_model,
        "nf_regnet_b3": get_nfnet_model,
        "nf_regnet_b4": get_nfnet_model,
        "nf_regnet_b5": get_nfnet_model,
        "nf_resnet26": get_nfnet_model,
        "nf_resnet50": get_nfnet_model,
        "nf_resnet101": get_nfnet_model,
        "nf_seresnet26": get_nfnet_model,
        "nf_seresnet50": get_nfnet_model,
        "nf_seresnet101": get_nfnet_model,
        "nf_ecaresnet26": get_nfnet_model,
        "nf_ecaresnet50": get_nfnet_model,
        "nf_ecaresnet101": get_nfnet_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

    return model_retrieval_functions[model_type](model_type, num_classes)


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
    model_retrieval_functions = {
        "pit_b_224": get_pit_model,
        "pit_s_224": get_pit_model,
        "pit_xs_224": get_pit_model,
        "pit_ti_224": get_pit_model,
        "pit_b_distilled_224": get_pit_model,
        "pit_s_distilled_224": get_pit_model,
        "pit_xs_distilled_224": get_pit_model,
        "pit_ti_distilled_224": get_pit_model,
        "pnasnet5large": get_pnasnet_model,
        "pvt_v2_b0": get_pvt_v2_model,
        "pvt_v2_b1": get_pvt_v2_model,
        "pvt_v2_b2": get_pvt_v2_model,
        "pvt_v2_b3": get_pvt_v2_model,
        "pvt_v2_b4": get_pvt_v2_model,
        "pvt_v2_b5": get_pvt_v2_model,
        "pvt_v2_b2_li": get_pvt_v2_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

    return model_retrieval_functions[model_type](model_type, num_classes)


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
    model_retrieval_functions = {
        "RegNet_X_400MF": get_regnet_model,
        "RegNet_X_800MF": get_regnet_model,
        "RegNet_X_1_6GF": get_regnet_model,
        "RegNet_X_3_2GF": get_regnet_model,
        "RegNet_X_16GF": get_regnet_model,
        "RegNet_Y_400MF": get_regnet_model,
        "RegNet_Y_800MF": get_regnet_model,
        "RegNet_Y_1_6GF": get_regnet_model,
        "RegNet_Y_3_2GF": get_regnet_model,
        "RegNet_Y_16GF": get_regnet_model,
        "regnetx_002": get_regnet_model,
        "regnetx_004": get_regnet_model,
        "regnetx_004_tv": get_regnet_model,
        "regnetx_006": get_regnet_model,
        "regnetx_008": get_regnet_model,
        "regnetx_016": get_regnet_model,
        "regnetx_032": get_regnet_model,
        "regnetx_040": get_regnet_model,
        "regnetx_064": get_regnet_model,
        "regnetx_080": get_regnet_model,
        "regnetx_120": get_regnet_model,
        "regnetx_160": get_regnet_model,
        "regnetx_320": get_regnet_model,
        "regnety_002": get_regnet_model,
        "regnety_004": get_regnet_model,
        "regnety_006": get_regnet_model,
        "regnety_008": get_regnet_model,
        "regnety_008_tv": get_regnet_model,
        "regnety_016": get_regnet_model,
        "regnety_032": get_regnet_model,
        "regnety_040": get_regnet_model,
        "regnety_064": get_regnet_model,
        "regnety_080": get_regnet_model,
        "regnety_080_tv": get_regnet_model,
        "regnety_120": get_regnet_model,
        "regnety_160": get_regnet_model,
        "regnety_320": get_regnet_model,
        "regnety_640": get_regnet_model,
        "regnety_1280": get_regnet_model,
        "regnety_2560": get_regnet_model,
        "regnety_040_sgn": get_regnet_model,
        "regnetv_040": get_regnet_model,
        "regnetv_064": get_regnet_model,
        "regnetz_005": get_regnet_model,
        "regnetz_040": get_regnet_model,
        "regnetz_040_h": get_regnet_model,
        "repghostnet_050": get_repghost_model,
        "repghostnet_058": get_repghost_model,
        "repghostnet_080": get_repghost_model,
        "repghostnet_100": get_repghost_model,
        "repghostnet_111": get_repghost_model,
        "repghostnet_130": get_repghost_model,
        "repghostnet_150": get_repghost_model,
        "repghostnet_200": get_repghost_model,
        "repvit_m1": get_repvit_model,
        "repvit_m2": get_repvit_model,
        "repvit_m3": get_repvit_model,
        "repvit_m0_9": get_repvit_model,
        "repvit_m1_0": get_repvit_model,
        "repvit_m1_1": get_repvit_model,
        "repvit_m1_5": get_repvit_model,
        "repvit_m2_3": get_repvit_model,
        "res2net50_26w_4s": get_res2net_model,
        "res2net101_26w_4s": get_res2net_model,
        "res2net50_26w_6s": get_res2net_model,
        "res2net50_26w_8s": get_res2net_model,
        "res2net50_48w_2s": get_res2net_model,
        "res2net50_14w_8s": get_res2net_model,
        "res2next50": get_res2net_model,
        "res2net50d": get_res2net_model,
        "res2net101d": get_res2net_model,
        "resnest14d": get_resnest_model,
        "resnest26d": get_resnest_model,
        "resnest50d": get_resnest_model,
        "resnest101e": get_resnest_model,
        "resnest200e": get_resnest_model,
        "resnest269e": get_resnest_model,
        "resnest50d_4s2x40d": get_resnest_model,
        "resnest50d_1s4x24d": get_resnest_model,
        "ResNet18": get_resnet_model,
        "ResNet34": get_resnet_model,
        "ResNet50": get_resnet_model,
        "ResNet101": get_resnet_model,
        "ResNet152": get_resnet_model,
        "resnet10t": get_resnet_model,
        "resnet14t": get_resnet_model,
        "resnet18": get_resnet_model,
        "resnet18d": get_resnet_model,
        "resnet34": get_resnet_model,
        "resnet34d": get_resnet_model,
        "resnet26": get_resnet_model,
        "resnet26t": get_resnet_model,
        "resnet26d": get_resnet_model,
        "resnet50": get_resnet_model,
        "resnet50c": get_resnet_model,
        "resnet50d": get_resnet_model,
        "resnet50s": get_resnet_model,
        "resnet50t": get_resnet_model,
        "resnet101": get_resnet_model,
        "resnet101c": get_resnet_model,
        "resnet101d": get_resnet_model,
        "resnet101s": get_resnet_model,
        "resnet152": get_resnet_model,
        "resnet152c": get_resnet_model,
        "resnet152d": get_resnet_model,
        "resnet152s": get_resnet_model,
        "resnet200": get_resnet_model,
        "resnet200d": get_resnet_model,
        "wide_resnet50_2": get_resnet_model,
        "wide_resnet101_2": get_resnet_model,
        "resnet50_gn": get_resnet_model,
        "resnext50_32x4d": get_resnet_model,
        "resnext50d_32x4d": get_resnet_model,
        "resnext101_32x4d": get_resnet_model,
        "resnext101_32x8d": get_resnet_model,
        "resnext101_32x16d": get_resnet_model,
        "resnext101_32x32d": get_resnet_model,
        "resnext101_64x4d": get_resnet_model,
        "ecaresnet26t": get_resnet_model,
        "ecaresnet50d": get_resnet_model,
        "ecaresnet50d_pruned": get_resnet_model,
        "ecaresnet50t": get_resnet_model,
        "ecaresnetlight": get_resnet_model,
        "ecaresnet101d": get_resnet_model,
        "ecaresnet101d_pruned": get_resnet_model,
        "ecaresnet200d": get_resnet_model,
        "ecaresnet269d": get_resnet_model,
        "ecaresnext26t_32x4d": get_resnet_model,
        "ecaresnext50t_32x4d": get_resnet_model,
        "seresnet18": get_resnet_model,
        "seresnet34": get_resnet_model,
        "seresnet50": get_resnet_model,
        "seresnet50t": get_resnet_model,
        "seresnet101": get_resnet_model,
        "seresnet152": get_resnet_model,
        "seresnet152d": get_resnet_model,
        "seresnet200d": get_resnet_model,
        "seresnet269d": get_resnet_model,
        "seresnext26d_32x4d": get_resnet_model,
        "seresnext26t_32x4d": get_resnet_model,
        "seresnext50_32x4d": get_resnet_model,
        "seresnext101_32x8d": get_resnet_model,
        "seresnext101d_32x8d": get_resnet_model,
        "seresnext101_64x4d": get_resnet_model,
        "senet154": get_resnet_model,
        "resnetblur18": get_resnet_model,
        "resnetblur50": get_resnet_model,
        "resnetblur50d": get_resnet_model,
        "resnetblur101d": get_resnet_model,
        "resnetaa34d": get_resnet_model,
        "resnetaa50": get_resnet_model,
        "resnetaa50d": get_resnet_model,
        "resnetaa101d": get_resnet_model,
        "seresnetaa50d": get_resnet_model,
        "seresnextaa101d_32x8d": get_resnet_model,
        "seresnextaa201d_32x8d": get_resnet_model,
        "resnetrs50": get_resnet_model,
        "resnetrs101": get_resnet_model,
        "resnetrs152": get_resnet_model,
        "resnetrs200": get_resnet_model,
        "resnetrs270": get_resnet_model,
        "resnetrs350": get_resnet_model,
        "resnetrs420": get_resnet_model,
        "resnetv2_50x1_bit": get_resnetv2_model,
        "resnetv2_50x3_bit": get_resnetv2_model,
        "resnetv2_101x1_bit": get_resnetv2_model,
        "resnetv2_101x3_bit": get_resnetv2_model,
        "resnetv2_152x2_bit": get_resnetv2_model,
        "resnetv2_152x4_bit": get_resnetv2_model,
        "resnetv2_50": get_resnetv2_model,
        "resnetv2_50d": get_resnetv2_model,
        "resnetv2_50t": get_resnetv2_model,
        "resnetv2_101": get_resnetv2_model,
        "resnetv2_101d": get_resnetv2_model,
        "resnetv2_152": get_resnetv2_model,
        "resnetv2_152d": get_resnetv2_model,
        "resnetv2_50d_gn": get_resnetv2_model,
        "resnetv2_50d_evos": get_resnetv2_model,
        "resnetv2_50d_frn": get_resnetv2_model,
        "ResNeXt50_32X4D": get_resnext_model,
        "ResNeXt101_32X8D": get_resnext_model,
        "ResNeXt101_64X4D": get_resnext_model,
        "rexnet_100": get_rexnet_model,
        "rexnet_130": get_rexnet_model,
        "rexnet_150": get_rexnet_model,
        "rexnet_200": get_rexnet_model,
        "rexnet_300": get_rexnet_model,
        "rexnetr_100": get_rexnet_model,
        "rexnetr_130": get_rexnet_model,
        "rexnetr_150": get_rexnet_model,
        "rexnetr_200": get_rexnet_model,
        "rexnetr_300": get_rexnet_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

    return model_retrieval_functions[model_type](model_type, num_classes)


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
    # Dictionary mapping model_type to retrieval functions
    model_retrieval_functions = {
        "selecsls42": get_selecsls_model,
        "selecsls42b": get_selecsls_model,
        "selecsls60": get_selecsls_model,
        "selecsls60b": get_selecsls_model,
        "selecsls84": get_selecsls_model,
        "legacy_seresnet18": get_senet_model,
        "legacy_seresnet34": get_senet_model,
        "legacy_seresnet50": get_senet_model,
        "legacy_seresnet101": get_senet_model,
        "legacy_seresnet152": get_senet_model,
        "legacy_senet154": get_senet_model,
        "legacy_seresnext26_32x4d": get_senet_model,
        "legacy_seresnext50_32x4d": get_senet_model,
        "legacy_seresnext101_32x4d": get_senet_model,
        "sequencer2d_s": get_sequencer_model,
        "sequencer2d_m": get_sequencer_model,
        "sequencer2d_l": get_sequencer_model,
        "ShuffleNet_V2_X0_5": get_shufflenet_model,
        "ShuffleNet_V2_X1_0": get_shufflenet_model,
        "ShuffleNet_V2_X1_5": get_shufflenet_model,
        "ShuffleNet_V2_X2_0": get_shufflenet_model,
        "skresnet18": get_sknet_model,
        "skresnet34": get_sknet_model,
        "skresnet50": get_sknet_model,
        "skresnet50d": get_sknet_model,
        "skresnext50_32x4d": get_sknet_model,
        "SqueezeNet1_0": get_squeezenet_model,
        "SqueezeNet1_1": get_squeezenet_model,
        "Swin_T": get_swin_transformer_model,
        "Swin_S": get_swin_transformer_model,
        "Swin_B": get_swin_transformer_model,
        "Swin_V2_T": get_swin_transformer_model,
        "Swin_V2_S": get_swin_transformer_model,
        "Swin_V2_B": get_swin_transformer_model,
        "swin_tiny_patch4_window7_224": get_swin_transformer_model,
        "swin_small_patch4_window7_224": get_swin_transformer_model,
        "swin_base_patch4_window7_224": get_swin_transformer_model,
        "swin_base_patch4_window12_384": get_swin_transformer_model,
        "swin_large_patch4_window7_224": get_swin_transformer_model,
        "swin_s3_tiny_224": get_swin_transformer_model,
        "swin_large_patch4_window12_384": get_swin_transformer_model,
        "swin_s3_small_224": get_swin_transformer_model,
        "swin_s3_base_224": get_swin_transformer_model,
        "swinv2_tiny_window16_256": get_swin_transformer_v2_model,
        "swinv2_tiny_window8_256": get_swin_transformer_v2_model,
        "swinv2_small_window16_256": get_swin_transformer_v2_model,
        "swinv2_small_window8_256": get_swin_transformer_v2_model,
        "swinv2_base_window16_256": get_swin_transformer_v2_model,
        "swinv2_base_window8_256": get_swin_transformer_v2_model,
        "swinv2_base_window12_192": get_swin_transformer_v2_model,
        "swinv2_base_window12to16_192to256": get_swin_transformer_v2_model,
        "swinv2_base_window12to24_192to384": get_swin_transformer_v2_model,
        "swinv2_large_window12_192": get_swin_transformer_v2_model,
        "swinv2_large_window12to16_192to256": get_swin_transformer_v2_model,
        "swinv2_large_window12to24_192to384": get_swin_transformer_v2_model,
        "swinv2_cr_tiny_384": get_swin_transformer_v2_cr_model,
        "swinv2_cr_tiny_224": get_swin_transformer_v2_cr_model,
        "swinv2_cr_tiny_ns_224": get_swin_transformer_v2_cr_model,
        "swinv2_cr_small_384": get_swin_transformer_v2_cr_model,
        "swinv2_cr_small_224": get_swin_transformer_v2_cr_model,
        "swinv2_cr_small_ns_224": get_swin_transformer_v2_cr_model,
        "swinv2_cr_small_ns_256": get_swin_transformer_v2_cr_model,
        "swinv2_cr_base_384": get_swin_transformer_v2_cr_model,
        "swinv2_cr_base_224": get_swin_transformer_v2_cr_model,
        "swinv2_cr_base_ns_224": get_swin_transformer_v2_cr_model,
        "swinv2_cr_large_384": get_swin_transformer_v2_cr_model,
        "swinv2_cr_large_224": get_swin_transformer_v2_cr_model,
        "swinv2_cr_huge_384": get_swin_transformer_v2_cr_model,
        "swinv2_cr_huge_224": get_swin_transformer_v2_cr_model,
        "swinv2_cr_giant_384": get_swin_transformer_v2_cr_model,
        "swinv2_cr_giant_224": get_swin_transformer_v2_cr_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

    return model_retrieval_functions[model_type](model_type, num_classes)


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
    # Dictionary mapping model_type to retrieval functions
    model_retrieval_functions = {
        "tiny_vit_5m_224": get_tiny_vit_model,
        "tiny_vit_11m_224": get_tiny_vit_model,
        "tiny_vit_21m_224": get_tiny_vit_model,
        "tiny_vit_21m_384": get_tiny_vit_model,
        "tiny_vit_21m_512": get_tiny_vit_model,
        "tnt_s_patch16_224": get_tnt_model,
        "tnt_b_patch16_224": get_tnt_model,
        "tresnet_m": get_tresnet_model,
        "tresnet_l": get_tresnet_model,
        "tresnet_xl": get_tresnet_model,
        "tresnet_v2_l": get_tresnet_model,
        "twins_pcpvt_small": get_twins_model,
        "twins_pcpvt_base": get_twins_model,
        "twins_pcpvt_large": get_twins_model,
        "twins_svt_small": get_twins_model,
        "twins_svt_base": get_twins_model,
        "twins_svt_large": get_twins_model,
    }
    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

    return model_retrieval_functions[model_type](model_type, num_classes)


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
    # Dictionary mapping model_type to retrieval functions
    model_retrieval_functions = {
        "VGG11": get_vgg_model,
        "VGG11_BN": get_vgg_model,
        "VGG13": get_vgg_model,
        "VGG13_BN": get_vgg_model,
        "VGG16": get_vgg_model,
        "VGG16_BN": get_vgg_model,
        "VGG19": get_vgg_model,
        "VGG19_BN": get_vgg_model,
        "vgg11": get_vgg_model,
        "vgg11_bn": get_vgg_model,
        "vgg13": get_vgg_model,
        "vgg13_bn": get_vgg_model,
        "vgg16": get_vgg_model,
        "vgg16_bn": get_vgg_model,
        "vgg19": get_vgg_model,
        "vgg19_bn": get_vgg_model,
        "visformer_tiny": get_visformer_model,
        "visformer_small": get_visformer_model,
        "ViT_B_16": get_vision_transformer_model,
        "ViT_B_32": get_vision_transformer_model,
        "ViT_L_16": get_vision_transformer_model,
        "ViT_L_32": get_vision_transformer_model,
        "ViT_H_14": get_vision_transformer_model,
        "vit_tiny_patch16_224": get_vision_transformer_model,
        "vit_tiny_patch16_384": get_vision_transformer_model,
        "vit_small_patch32_224": get_vision_transformer_model,
        "vit_small_patch32_384": get_vision_transformer_model,
        "vit_small_patch16_224": get_vision_transformer_model,
        "vit_small_patch16_384": get_vision_transformer_model,
        "vit_small_patch8_224": get_vision_transformer_model,
        "vit_base_patch32_224": get_vision_transformer_model,
        "vit_base_patch32_384": get_vision_transformer_model,
        "vit_base_patch16_224": get_vision_transformer_model,
        "vit_base_patch16_384": get_vision_transformer_model,
        "vit_base_patch8_224": get_vision_transformer_model,
        "vit_large_patch32_224": get_vision_transformer_model,
        "vit_large_patch32_384": get_vision_transformer_model,
        "vit_large_patch16_224": get_vision_transformer_model,
        "vit_large_patch16_384": get_vision_transformer_model,
        "vit_large_patch14_224": get_vision_transformer_model,
        "vit_giant_patch14_224": get_vision_transformer_model,
        "vit_gigantic_patch14_224": get_vision_transformer_model,
        "vit_base_patch16_224_miil": get_vision_transformer_model,
        "vit_medium_patch16_gap_240": get_vision_transformer_model,
        "vit_medium_patch16_gap_256": get_vision_transformer_model,
        "vit_medium_patch16_gap_384": get_vision_transformer_model,
        "vit_base_patch16_gap_224": get_vision_transformer_model,
        "vit_huge_patch14_gap_224": get_vision_transformer_model,
        "vit_huge_patch16_gap_448": get_vision_transformer_model,
        "vit_giant_patch16_gap_224": get_vision_transformer_model,
        "vit_xsmall_patch16_clip_224": get_vision_transformer_model,
        "vit_medium_patch32_clip_224": get_vision_transformer_model,
        "vit_medium_patch16_clip_224": get_vision_transformer_model,
        "vit_betwixt_patch32_clip_224": get_vision_transformer_model,
        "vit_base_patch32_clip_224": get_vision_transformer_model,
        "vit_base_patch32_clip_256": get_vision_transformer_model,
        "vit_base_patch32_clip_384": get_vision_transformer_model,
        "vit_base_patch32_clip_448": get_vision_transformer_model,
        "vit_base_patch16_clip_224": get_vision_transformer_model,
        "vit_base_patch16_clip_384": get_vision_transformer_model,
        "vit_large_patch14_clip_224": get_vision_transformer_model,
        "vit_large_patch14_clip_336": get_vision_transformer_model,
        "vit_huge_patch14_clip_224": get_vision_transformer_model,
        "vit_huge_patch14_clip_336": get_vision_transformer_model,
        "vit_huge_patch14_clip_378": get_vision_transformer_model,
        "vit_giant_patch14_clip_224": get_vision_transformer_model,
        "vit_gigantic_patch14_clip_224": get_vision_transformer_model,
        "vit_base_patch32_clip_quickgelu_224": get_vision_transformer_model,
        "vit_base_patch16_clip_quickgelu_224": get_vision_transformer_model,
        "vit_large_patch14_clip_quickgelu_224": get_vision_transformer_model,
        "vit_large_patch14_clip_quickgelu_336": get_vision_transformer_model,
        "vit_huge_patch14_clip_quickgelu_224": get_vision_transformer_model,
        "vit_huge_patch14_clip_quickgelu_378": get_vision_transformer_model,
        "vit_base_patch32_plus_256": get_vision_transformer_model,
        "vit_base_patch16_plus_240": get_vision_transformer_model,
        "vit_base_patch16_rpn_224": get_vision_transformer_model,
        "vit_small_patch16_36x1_224": get_vision_transformer_model,
        "vit_small_patch16_18x2_224": get_vision_transformer_model,
        "vit_base_patch16_18x2_224": get_vision_transformer_model,
        "eva_large_patch14_196": get_vision_transformer_model,
        "eva_large_patch14_336": get_vision_transformer_model,
        "flexivit_small": get_vision_transformer_model,
        "flexivit_base": get_vision_transformer_model,
        "flexivit_large": get_vision_transformer_model,
        "vit_base_patch16_xp_224": get_vision_transformer_model,
        "vit_large_patch14_xp_224": get_vision_transformer_model,
        "vit_huge_patch14_xp_224": get_vision_transformer_model,
        "vit_small_patch14_dinov2": get_vision_transformer_model,
        "vit_base_patch14_dinov2": get_vision_transformer_model,
        "vit_large_patch14_dinov2": get_vision_transformer_model,
        "vit_giant_patch14_dinov2": get_vision_transformer_model,
        "vit_small_patch14_reg4_dinov2": get_vision_transformer_model,
        "vit_base_patch14_reg4_dinov2": get_vision_transformer_model,
        "vit_large_patch14_reg4_dinov2": get_vision_transformer_model,
        "vit_giant_patch14_reg4_dinov2": get_vision_transformer_model,
        "vit_base_patch16_siglip_224": get_vision_transformer_model,
        "vit_base_patch16_siglip_256": get_vision_transformer_model,
        "vit_base_patch16_siglip_384": get_vision_transformer_model,
        "vit_base_patch16_siglip_512": get_vision_transformer_model,
        "vit_large_patch16_siglip_256": get_vision_transformer_model,
        "vit_large_patch16_siglip_384": get_vision_transformer_model,
        "vit_so400m_patch14_siglip_224": get_vision_transformer_model,
        "vit_so400m_patch14_siglip_384": get_vision_transformer_model,
        "vit_base_patch16_siglip_gap_224": get_vision_transformer_model,
        "vit_base_patch16_siglip_gap_256": get_vision_transformer_model,
        "vit_base_patch16_siglip_gap_384": get_vision_transformer_model,
        "vit_base_patch16_siglip_gap_512": get_vision_transformer_model,
        "vit_large_patch16_siglip_gap_256": get_vision_transformer_model,
        "vit_large_patch16_siglip_gap_384": get_vision_transformer_model,
        "vit_so400m_patch14_siglip_gap_224": get_vision_transformer_model,
        "vit_so400m_patch14_siglip_gap_384": get_vision_transformer_model,
        "vit_so400m_patch14_siglip_gap_448": get_vision_transformer_model,
        "vit_so400m_patch14_siglip_gap_896": get_vision_transformer_model,
        "vit_wee_patch16_reg1_gap_256": get_vision_transformer_model,
        "vit_pwee_patch16_reg1_gap_256": get_vision_transformer_model,
        "vit_little_patch16_reg4_gap_256": get_vision_transformer_model,
        "vit_medium_patch16_reg1_gap_256": get_vision_transformer_model,
        "vit_medium_patch16_reg4_gap_256": get_vision_transformer_model,
        "vit_mediumd_patch16_reg4_gap_256": get_vision_transformer_model,
        "vit_betwixt_patch16_reg1_gap_256": get_vision_transformer_model,
        "vit_betwixt_patch16_reg4_gap_256": get_vision_transformer_model,
        "vit_base_patch16_reg4_gap_256": get_vision_transformer_model,
        "vit_so150m_patch16_reg4_map_256": get_vision_transformer_model,
        "vit_so150m_patch16_reg4_gap_256": get_vision_transformer_model,
        "vit_tiny_r_s16_p8_224": get_vision_transformer_hybrid_model,
        "vit_tiny_r_s16_p8_384": get_vision_transformer_hybrid_model,
        "vit_small_r26_s32_224": get_vision_transformer_hybrid_model,
        "vit_small_r26_s32_384": get_vision_transformer_hybrid_model,
        "vit_base_r26_s32_224": get_vision_transformer_hybrid_model,
        "vit_base_r50_s16_224": get_vision_transformer_hybrid_model,
        "vit_base_r50_s16_384": get_vision_transformer_hybrid_model,
        "vit_large_r50_s32_224": get_vision_transformer_hybrid_model,
        "vit_large_r50_s32_384": get_vision_transformer_hybrid_model,
        "vit_small_resnet26d_224": get_vision_transformer_hybrid_model,
        "vit_small_resnet50d_s16_224": get_vision_transformer_hybrid_model,
        "vit_base_resnet26d_224": get_vision_transformer_hybrid_model,
        "vit_base_resnet50d_224": get_vision_transformer_hybrid_model,
        "vit_relpos_base_patch32_plus_rpn_256": get_vision_transformer_relpos_model,
        "vit_relpos_base_patch16_plus_240": get_vision_transformer_relpos_model,
        "vit_relpos_small_patch16_224": get_vision_transformer_relpos_model,
        "vit_relpos_medium_patch16_224": get_vision_transformer_relpos_model,
        "vit_relpos_base_patch16_224": get_vision_transformer_relpos_model,
        "vit_srelpos_small_patch16_224": get_vision_transformer_relpos_model,
        "vit_srelpos_medium_patch16_224": get_vision_transformer_relpos_model,
        "vit_relpos_medium_patch16_cls_224": get_vision_transformer_relpos_model,
        "vit_relpos_base_patch16_cls_224": get_vision_transformer_relpos_model,
        "vit_relpos_base_patch16_clsgap_224": get_vision_transformer_relpos_model,
        "vit_relpos_small_patch16_rpn_224": get_vision_transformer_relpos_model,
        "vit_relpos_medium_patch16_rpn_224": get_vision_transformer_relpos_model,
        "vit_relpos_base_patch16_rpn_224": get_vision_transformer_relpos_model,
        "samvit_base_patch16": get_vision_transformer_sam_model,
        "samvit_large_patch16": get_vision_transformer_sam_model,
        "samvit_huge_patch16": get_vision_transformer_sam_model,
        "samvit_base_patch16_224": get_vision_transformer_sam_model,
        "vitamin_small": get_vitamin_model,
        "vitamin_base": get_vitamin_model,
        "vitamin_large": get_vitamin_model,
        "vitamin_large_256": get_vitamin_model,
        "vitamin_large_336": get_vitamin_model,
        "vitamin_large_384": get_vitamin_model,
        "vitamin_xlarge_256": get_vitamin_model,
        "vitamin_xlarge_336": get_vitamin_model,
        "vitamin_xlarge_384": get_vitamin_model,
        "volo_d1_224": get_volo_model,
        "volo_d1_384": get_volo_model,
        "volo_d2_224": get_volo_model,
        "volo_d2_384": get_volo_model,
        "volo_d3_224": get_volo_model,
        "volo_d3_448": get_volo_model,
        "volo_d4_224": get_volo_model,
        "volo_d4_448": get_volo_model,
        "volo_d5_224": get_volo_model,
        "volo_d5_448": get_volo_model,
        "volo_d5_512": get_volo_model,
        "vovnet39a": get_vovnet_model,
        "vovnet57a": get_vovnet_model,
        "ese_vovnet19b_slim_dw": get_vovnet_model,
        "ese_vovnet19b_dw": get_vovnet_model,
        "ese_vovnet19b_slim": get_vovnet_model,
        "ese_vovnet39b": get_vovnet_model,
        "ese_vovnet57b": get_vovnet_model,
        "ese_vovnet99b": get_vovnet_model,
        "eca_vovnet39b": get_vovnet_model,
        "ese_vovnet39b_evos": get_vovnet_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

    return model_retrieval_functions[model_type](model_type, num_classes)


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
    model_retrieval_functions = {
        "Wide_ResNet50_2": get_wide_resnet_model,
        "Wide_ResNet101_2": get_wide_resnet_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

    return model_retrieval_functions[model_type](model_type, num_classes)


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
    model_retrieval_functions = {
        "legacy_xception": get_xception_model,
        "xception41": get_xception_model,
        "xception65": get_xception_model,
        "xception71": get_xception_model,
        "xception41p": get_xception_model,
        "xception65p": get_xception_model,
        "xcit_nano_12_p16_224": get_xcit_model,
        "xcit_nano_12_p16_384": get_xcit_model,
        "xcit_tiny_12_p16_224": get_xcit_model,
        "xcit_tiny_12_p16_384": get_xcit_model,
        "xcit_small_12_p16_224": get_xcit_model,
        "xcit_small_12_p16_384": get_xcit_model,
        "xcit_tiny_24_p16_224": get_xcit_model,
        "xcit_tiny_24_p16_384": get_xcit_model,
        "xcit_small_24_p16_224": get_xcit_model,
        "xcit_small_24_p16_384": get_xcit_model,
        "xcit_medium_24_p16_224": get_xcit_model,
        "xcit_medium_24_p16_384": get_xcit_model,
        "xcit_large_24_p16_224": get_xcit_model,
        "xcit_large_24_p16_384": get_xcit_model,
        "xcit_nano_12_p8_224": get_xcit_model,
        "xcit_nano_12_p8_384": get_xcit_model,
        "xcit_tiny_12_p8_224": get_xcit_model,
        "xcit_tiny_12_p8_384": get_xcit_model,
        "xcit_small_12_p8_224": get_xcit_model,
        "xcit_small_12_p8_384": get_xcit_model,
        "xcit_tiny_24_p8_224": get_xcit_model,
        "xcit_tiny_24_p8_384": get_xcit_model,
        "xcit_small_24_p8_224": get_xcit_model,
        "xcit_small_24_p8_384": get_xcit_model,
        "xcit_medium_24_p8_224": get_xcit_model,
        "xcit_medium_24_p8_384": get_xcit_model,
        "xcit_large_24_p8_224": get_xcit_model,
        "xcit_large_24_p8_384": get_xcit_model,
    }

    if model_type not in model_retrieval_functions:
        raise ValueError(f"Unknown model_type provided: {model_type}")

    return model_retrieval_functions[model_type](model_type, num_classes)


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
        "a": get_family_model_a,
        "b": get_family_model_b,
        "c": get_family_model_c,
        "e": get_family_model_e,
        "f": get_family_model_f,
        "g": get_family_model_g,
        "h": get_family_model_h,
        "i": get_family_model_i,
        "l": get_family_model_l,
        "m": get_family_model_m,
        "n": get_family_model_n,
        "p": get_family_model_p,
        "r": get_family_model_r,
        "s": get_family_model_s,
        "t": get_family_model_t,
        "v": get_family_model_v,
        "w": get_family_model_w,
        "x": get_family_model_x,
    }

    # Convert the first letter of the model_type to lowercase
    first_letter = model_type[0].lower()

    # Default value if no matching case is found
    model = "Error"

    # Retrieve the corresponding function and call it
    if first_letter in model_functions:
        model = model_functions[first_letter](model_type, num_classes)

    return model
