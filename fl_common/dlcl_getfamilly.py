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
    Retourne un modèle appartenant à l'une des familles de modèles spécifiées, selon le type spécifié.

    Args:
        model_type (str): Le type de modèle à récupérer. Doit être l'un des suivants:

            Pour les modèles BEiT (Bottleneck Transformers):
            - 'beit_base_patch16_224'
            - 'beit_base_patch16_384'
            - 'beit_large_patch16_224'
            - 'beit_large_patch16_384'
            - 'beit_large_patch16_512'
            - 'beitv2_base_patch16_224'
            - 'beitv2_large_patch16_224'

            Pour les modèles Byoanet :
            - 'botnet26t_256'
            - 'sebotnet33ts_256'
            - 'botnet50ts_256'
            - 'eca_botnext26ts_256'
            - 'halonet_h1'
            - 'halonet26t'
            - 'sehalonet33ts'
            - 'halonet50ts'
            - 'eca_halonext26ts'
            - 'lambda_resnet26t'
            - 'lambda_resnet50ts'
            - 'lambda_resnet26rpt_256'
            - 'haloregnetz_b'
            - 'lamhalobotnet50ts_256'
            - 'halo2botnet50ts_256'

            Pour les modèles Byobnet :
            - 'gernet_l'
            - 'gernet_m'
            - 'gernet_s'
            - 'repvgg_a0'
            - 'repvgg_a1'
            - 'repvgg_a2'
            - 'repvgg_b0'
            - 'repvgg_b1'
            - 'repvgg_b1g4'
            - 'repvgg_b2'
            - 'repvgg_b2g4'
            - 'repvgg_b3'
            - 'repvgg_b3g4'
            - 'repvgg_d2se'
            - 'resnet51q'
            - 'resnet61q'
            - 'resnext26ts'
            - 'gcresnext26ts'
            - 'seresnext26ts'
            - 'eca_resnext26ts'
            - 'bat_resnext26ts'
            - 'resnet32ts'
            - 'resnet33ts'
            - 'gcresnet33ts'
            - 'seresnet33ts'
            - 'eca_resnet33ts'
            - 'gcresnet50t'
            - 'gcresnext50ts'

        num_classes (int): Le nombre de classes pour la tâche de classification.

    Returns:
        torch.nn.Module: Le modèle correspondant au type spécifié.

    Raises:
        ValueError: Si le type de modèle spécifié n'est pas reconnu.
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
                        'seresnet33ts', 'eca_resnet33ts', 'gcresnet50t', 'gcresnext50ts']:
        model = get_byobnet_model(model_type, num_classes)

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
    elif model_type in ['DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201', 'densenet121', 'densenetblur121d',
                        'densenet169', 'densenet201', 'densenet161', 'densenet264d']:
        model = get_densenet_model(model_type, num_classes)
    elif model_type in ['dla60_res2net', 'dla60_res2next', 'dla34', 'dla46_c', 'dla46x_c', 'dla60x_c', 'dla60',
                        'dla60x', 'dla102', 'dla102x', 'dla102x2', 'dla169']:
        model = get_dla_model(model_type, num_classes)
    elif model_type in ['dpn48b', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107']:
        model = get_dpn_model(model_type, num_classes)

    return model


def get_family_model_e(model_type, num_classes):
    """
    Get a model of the specified type.

    Parameters:
        model_type (str): Type of model to retrieve.
                          Choices:
                              - For EdgeNext: 'edgenext_xx_small', 'edgenext_x_small', 'edgenext_small',
                                              'edgenext_base', 'edgenext_small_rw'.
                              - For Efficientformer: 'efficientformer_l1', 'efficientformer_l3', 'efficientformer_l7'.
                              - For EfficientNet: 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2',
                                                  'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
                                                  'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2S',
                                                  'EfficientNetV2M', 'EfficientNetV2L'.
                              - For Eva: 'eva_giant_patch14_224', 'eva_giant_patch14_336', 'eva_giant_patch14_560',
                                         'eva02_tiny_patch14_224', 'eva02_small_patch14_224', 'eva02_base_patch14_224',
                                         'eva02_large_patch14_224', 'eva02_tiny_patch14_336', 'eva02_small_patch14_336',
                                         'eva02_base_patch14_448', 'eva02_large_patch14_448',
                                         'eva_giant_patch14_clip_224', 'eva02_base_patch16_clip_224',
                                         'eva02_large_patch14_clip_224', 'eva02_large_patch14_clip_336',
                                         'eva02_enormous_patch14_clip_224'.
        num_classes (int): Number of classes for the classification task.

    Returns:
        model: The requested model instance.

    Raises:
        ValueError: If the specified model_type is not one of the supported architectures.
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
    Create and return an instance of a specified deep learning model architecture.

    Args:
    - model_type (str): The type of model architecture to create. It should be one of the following:
        - 'MaxVit_T' for MaxVit architecture.
        - 'MNASNet0_5', 'MNASNet0_75', 'MNASNet1_0', 'MNASNet1_3' for MNASNet architectures.
        - 'poolformer_s12', 'poolformer_s24', 'poolformer_s36', 'poolformer_m36', 'poolformer_m48',
          'poolformerv2_s12', 'poolformerv2_s24', 'poolformerv2_s36', 'poolformerv2_m36',
          'poolformerv2_m48', 'convformer_s18', 'convformer_s36', 'convformer_m36', 'convformer_b36',
          'caformer_s18', 'caformer_s36', 'caformer_m36', 'caformer_b36' for Poolformer, Convformer,
          CAformer architectures.
        - 'mixer_s32_224', 'mixer_s16_224', 'mixer_b32_224', 'mixer_b16_224', 'mixer_l32_224',
          'mixer_l16_224', 'gmixer_12_224', 'gmixer_24_224', 'resmlp_12_224', 'resmlp_24_224',
          'resmlp_36_224', 'resmlp_big_24_224', 'gmlp_ti16_224', 'gmlp_s16_224', 'gmlp_b16_224'
          for MLP-Mixer architectures.
        - 'MobileNet_V2', 'MobileNet_V3_Small', 'MobileNet_V3_Large' for MobileNet architectures.
        - 'mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s', 'mobilevitv2_050', 'mobilevitv2_075',
          'mobilevitv2_100', 'mobilevitv2_125', 'mobilevitv2_150', 'mobilevitv2_175', 'mobilevitv2_200'
          for MobileViT architectures.
        - 'mvitv2_tiny', 'mvitv2_small', 'mvitv2_base', 'mvitv2_large', 'mvitv2_small_cls',
          'mvitv2_base_cls', 'mvitv2_large_cls', 'mvitv2_huge_cls' for MViTv2 architectures.
    - num_classes (int): The number of output classes for the model.

    Returns:
    - torch.nn.Module: An instance of the specified deep learning model architecture.

    Raises:
    - ValueError: If an unknown model architecture type is specified.
    """
    model = "Unknown"

    if model_type in ['MaxVit_T']:
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
    Get a model from the 'N' family based on the specified model type.

    Args:
        model_type (str): The type of model from the 'N' family. It can be one of the following:
            - 'nasnetalarge': NASNet-A Large model.
            - 'nest_base', 'nest_small', 'nest_tiny', 'nest_base_jx', 'nest_small_jx', 'nest_tiny_jx':
              Various NEST architectures.
            - 'nextvit_small', 'nextvit_base', 'nextvit_large': NEXTVIT architectures.
            - 'dm_nfnet_f0' to 'nf_ecaresnet101': Different architectures from the NFNet family.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The selected model from the 'N' family.

    Raises:
        ValueError: If an unknown model type is specified.
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
    Get a model from the family of models including PIT, PNASNet, and PVTv2.

    Args:
        model_type (str): The type of the model architecture.
            - For PIT: 'pit_b_224', 'pit_s_224', 'pit_xs_224', 'pit_ti_224', 'pit_b_distilled_224',
              'pit_s_distilled_224', 'pit_xs_distilled_224', 'pit_ti_distilled_224'
            - For PNASNet: 'pnasnet5large'
            - For PVTv2: 'pvt_v2_b0', 'pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5', 'pvt_v2_b2_li'
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The requested model.

    Raises:
        ValueError: If the specified model_type is unknown.
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
    Returns a model based on the provided model type and number of classes.

    Args:
    - model_type (str): The type of model. It should be one of the following:
        - For RegNet models: 'RegNet_X_400MF', 'RegNet_X_800MF', 'RegNet_X_1_6GF', 'RegNet_X_3_2GF', 'RegNet_X_16GF',
                             'RegNet_Y_400MF', 'RegNet_Y_800MF', 'RegNet_Y_1_6GF', 'RegNet_Y_3_2GF', 'RegNet_Y_16GF'
        - For RepGhost models: 'repghostnet_050', 'repghostnet_058', 'repghostnet_080', 'repghostnet_100', 'repghostnet_111',
                               'repghostnet_130', 'repghostnet_150', 'repghostnet_200'
        - For RepVIT models: 'repvit_m1', 'repvit_m2', 'repvit_m3', 'repvit_m0_9', 'repvit_m1_0', 'repvit_m1_1',
                              'repvit_m1_5', 'repvit_m2_3'
        - For Res2Net models: 'res2net50_26w_4s', 'res2net101_26w_4s', 'res2net50_26w_6s', 'res2net50_26w_8s',
                               'res2net50_48w_2s', 'res2net50_14w_8s', 'res2next50', 'res2net50d', 'res2net101d'
        - For ResNet models: 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNeXt50_32X4D',
                              'ResNeXt101_32X4D', 'ResNeXt101_64X4D'
        - For ResNeXt models: 'ResNeXt50_32X4D', 'ResNeXt101_32X8D', 'ResNeXt101_64X4D'
    - num_classes (int): The number of output classes for the model.

    Returns:
    - model: The model instantiated based on the specified architecture.

    Raises:
    - ValueError: If the provided model_type is not recognized.
    """
    model = "Unknown"

    if model_type in ['RegNet_X_400MF', 'RegNet_X_800MF', 'RegNet_X_1_6GF', 'RegNet_X_3_2GF', 'RegNet_X_16GF',
                      'RegNet_Y_400MF', 'RegNet_Y_800MF', 'RegNet_Y_1_6GF', 'RegNet_Y_3_2GF', 'RegNet_Y_16GF']:
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
    Create and return an instance of a specified deep learning model architecture.

    Args:
    - model_type (str): The type of model architecture to create. It should be one of the following:
        - 'selecsls42', 'selecsls42b', 'selecsls60', 'selecsls60b', 'selecsls84' for SelecSLS architectures.
        - 'legacy_seresnet18', 'legacy_seresnet34', 'legacy_seresnet50', 'legacy_seresnet101',
          'legacy_seresnet152', 'legacy_senet154', 'legacy_seresnext26_32x4d',
          'legacy_seresnext50_32x4d', 'legacy_seresnext101_32x4d' for Legacy SEResNet and SENet architectures.
        - 'sequencer2d_s', 'sequencer2d_m', 'sequencer2d_l' for Sequencer2D architectures.
        - 'ShuffleNet_V2_X0_5', 'ShuffleNet_V2_X1_0', 'ShuffleNet_V2_X1_5', 'ShuffleNet_V2_X2_0'
          for ShuffleNetV2 architectures.
        - 'skresnet18', 'skresnet34', 'skresnet50', 'skresnet50d', 'skresnext50_32x4d' for SKNet architectures.
        - 'SqueezeNet1_0', 'SqueezeNet1_1' for SqueezeNet architectures.
        - 'Swin_T', 'Swin_S', 'Swin_B', 'Swin_V2_T', 'Swin_V2_S', 'Swin_V2_B' for Swin Transformer architectures.
    - num_classes (int): The number of output classes for the model.

    Returns:
    - torch.nn.Module: An instance of the specified deep learning model architecture.

    Raises:
    - ValueError: If an unknown model architecture type is specified.
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
    Create and return an instance of a specified deep learning model architecture.

    Args:
    - model_type (str): The type of model architecture to create. It should be one of the following:
        - 'tiny_vit_5m_224', 'tiny_vit_11m_224', 'tiny_vit_21m_224', 'tiny_vit_21m_384', 'tiny_vit_21m_512'
        for TinyViT architectures.
        - 'tnt_s_patch16_224', 'tnt_b_patch16_224' for TNT architectures.
        - 'tresnet_m', 'tresnet_l', 'tresnet_xl', 'tresnet_v2_l' for TResNet architectures.
        - 'twins_pcpvt_small', 'twins_pcpvt_base', 'twins_pcpvt_large', 'twins_svt_small',
          'twins_svt_base', 'twins_svt_large' for Twins architectures.
    - num_classes (int): The number of output classes for the model.

    Returns:
    - torch.nn.Module: An instance of the specified deep learning model architecture.

    Raises:
    - ValueError: If an unknown model architecture type is specified.
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
                        'xcit_small_12_p16_224', 'xcit_small_12_p16_384', 'xcit_tiny_24_p16_224',
                        'xcit_tiny_24_p16_384',
                        'xcit_small_24_p16_224', 'xcit_small_24_p16_384', 'xcit_medium_24_p16_224',
                        'xcit_medium_24_p16_384', 'xcit_large_24_p16_224', 'xcit_large_24_p16_384',
                        'xcit_nano_12_p8_224',
                        'xcit_nano_12_p8_384', 'xcit_tiny_12_p8_224', 'xcit_tiny_12_p8_384', 'xcit_small_12_p8_224',
                        'xcit_small_12_p8_384', 'xcit_tiny_24_p8_224', 'xcit_tiny_24_p8_384', 'xcit_small_24_p8_224',
                        'xcit_small_24_p8_384', 'xcit_medium_24_p8_224', 'xcit_medium_24_p8_384',
                        'xcit_large_24_p8_224',
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
        case 'n':
            model = get_family_model_n(model_type, num_classes)
        case 'p':
            model = get_family_model_p(model_type, num_classes)
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
