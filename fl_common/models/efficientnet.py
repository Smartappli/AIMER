import torch.nn as nn
from torchvision import models
from timm import create_model


def get_efficientnet_model(efficientnet_type, num_classes):
    """
    Obtain an EfficientNet model with a specified architecture type and modify it for the given number of classes.

    Args:
    - efficientnet_type (str): Type of EfficientNet architecture to be loaded.
      Options: 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
               'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2S', 'EfficientNetV2M',
               'EfficientNetV2L', 'mnasnet_050', 'mnasnet_075', 'mnasnet_100', 'mnasnet_140', 'semnasnet_050',
               'semnasnet_075', 'semnasnet_100', 'semnasnet_140', 'mnasnet_small', 'mobilenetv2_035', 'mobilenetv2_050',
               'mobilenetv2_075', 'mobilenetv2_100', 'mobilenetv2_140', 'mobilenetv2_110d', 'mobilenetv2_120d',
               'fbnetc_100', 'spnasnet_100', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
               'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_b8',
               'efficientnet_l2', 'efficientnet_b0_gn', 'efficientnet_b0_g8_gn', 'efficientnet_b0_g16_evos',
               'efficientnet_b3_gn', 'efficientnet_b3_g8_gn', 'efficientnet_es', 'efficientnet_es_pruned',
               'efficientnet_em', 'efficientnet_el', 'efficientnet_el_pruned', 'efficientnet_cc_b0_4e',
               'efficientnet_cc_b0_8e', 'efficientnet_cc_b1_8e', 'efficientnet_lite0', 'efficientnet_lite1',
               'efficientnet_lite2', 'efficientnet_lite3', 'efficientnet_lite4', 'efficientnet_b1_pruned',
               'efficientnet_b2_pruned', 'efficientnet_b3_pruned', 'efficientnetv2_rw_t', 'gc_efficientnetv2_rw_t',
               'efficientnetv2_rw_s', 'efficientnetv2_rw_m', 'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l',
               'efficientnetv2_xl', 'tf_efficientnet_b0', 'tf_efficientnet_b1', 'tf_efficientnet_b2',
               'tf_efficientnet_b3', 'tf_efficientnet_b4', 'tf_efficientnet_b5', 'tf_efficientnet_b6',
               'tf_efficientnet_b7', 'tf_efficientnet_b8', 'tf_efficientnet_l2', 'tf_efficientnet_es',
               'tf_efficientnet_em', 'tf_efficientnet_el', 'tf_efficientnet_cc_b0_4e', 'tf_efficientnet_cc_b0_8e',
               'tf_efficientnet_cc_b1_8e', 'tf_efficientnet_lite0', 'tf_efficientnet_lite1', 'tf_efficientnet_lite2',
               'tf_efficientnet_lite3', 'tf_efficientnet_lite4', 'tf_efficientnetv2_s', 'tf_efficientnetv2_m',
               'tf_efficientnetv2_l', 'tf_efficientnetv2_xl', 'tf_efficientnetv2_b0', 'tf_efficientnetv2_b1',
               'tf_efficientnetv2_b2', 'tf_efficientnetv2_b3', 'mixnet_s', 'mixnet_m', 'mixnet_l', 'mixnet_xl',
               'mixnet_xxl', 'tf_mixnet_s', 'tf_mixnet_m', 'tf_mixnet_l', 'tinynet_a', 'tinynet_b', 'tinynet_c',
               'tinynet_d', 'tinynet_e'
    - num_classes (int): Number of output classes for the modified model. Default is 1000.

    Returns:
    - efficientnet_model (torch.nn.Module): The modified EfficientNet model.

    Raises:
    - ValueError: If the provided efficientnet_type is not recognized.

    Note:
    - This function loads a pre-trained EfficientNet model and modifies its last fully connected layer
      to match the specified number of output classes.

    Example Usage:
    ```python
    # Obtain an EfficientNetB0 model with 10 output classes
    model = get_efficientnet_model(efficientnet_type='EfficientNetB0', num_classes=10)
    ```
    """
    torch_vision = False
    # Load the pre-trained version of EfficientNet
    if efficientnet_type == 'EfficientNetB0':
        torch_vision = True
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            efficientnet_model = models.efficientnet_b0(weights=weights)
        except ValueError:
            efficientnet_model = models.efficientnet_b0(weights=None)
    elif efficientnet_type == 'EfficientNetB1':
        torch_vision = True
        try:
            weights = models.EfficientNet_B1_Weights.DEFAULT
            efficientnet_model = models.efficientnet_b1(weights=weights)
        except ValueError:
            efficientnet_model = models.efficientnet_b1(weights=None)
    elif efficientnet_type == 'EfficientNetB2':
        torch_vision = True
        try:
            weights = models.EfficientNet_B2_Weights.DEFAULT
            efficientnet_model = models.efficientnet_b2(weights=weights)
        except ValueError:
            efficientnet_model = models.efficientnet_b2(weights=None)
    elif efficientnet_type == 'EfficientNetB3':
        torch_vision = True
        try:
            weights = models.EfficientNet_B3_Weights.DEFAULT
            efficientnet_model = models.efficientnet_b3(weights=weights)
        except ValueError:
            efficientnet_model = models.efficientnet_b3(weights=None)
    elif efficientnet_type == 'EfficientNetB4':
        torch_vision = True
        try:
            weights = models.EfficientNet_B4_Weights.DEFAULT
            efficientnet_model = models.efficientnet_b4(weights=weights)
        except ValueError:
            efficientnet_model = models.efficientnet_b4(weights=None)
    elif efficientnet_type == 'EfficientNetB5':
        torch_vision = True
        try:
            weights = models.EfficientNet_B5_Weights.DEFAULT
            efficientnet_model = models.efficientnet_b5(weights=weights)
        except ValueError:
            efficientnet_model = models.efficientnet_b5(weights=None)
    elif efficientnet_type == 'EfficientNetB6':
        torch_vision = True
        try:
            weights = models.EfficientNet_B6_Weights.DEFAULT
            efficientnet_model = models.efficientnet_b6(weights=weights)
        except ValueError:
            efficientnet_model = models.efficientnet_b6(weights=None)
    elif efficientnet_type == 'EfficientNetB7':
        torch_vision = True
        try:
            weights = models.EfficientNet_B7_Weights.DEFAULT
            efficientnet_model = models.efficientnet_b7(weights=weights)
        except ValueError:
            efficientnet_model = models.efficientnet_b7(weights=None)
    elif efficientnet_type == 'EfficientNetV2S':
        torch_vision = True
        try:
            weights = models.EfficientNet_V2_S_Weights.DEFAULT
            efficientnet_model = models.efficientnet_v2_s(weights=weights)
        except ValueError:
            efficientnet_model = models.efficientnet_v2_s(weights=None)
    elif efficientnet_type == 'EfficientNetV2M':
        torch_vision = True
        try:
            weights = models.EfficientNet_V2_M_Weights.DEFAULT
            efficientnet_model = models.efficientnet_v2_m(weights=weights)
        except ValueError:
            efficientnet_model = models.efficientnet_v2_m(weights=None)
    elif efficientnet_type == 'EfficientNetV2L':
        torch_vision = True
        try:
            weights = models.EfficientNet_V2_L_Weights.DEFAULT
            efficientnet_model = models.efficientnet_v2_l(weights=weights)
        except ValueError:
            efficientnet_model = models.efficientnet_v2_l(weights=None)
    elif efficientnet_type == "mnasnet_050":
        try:
            efficientnet_model = create_model('mnasnet_050',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mnasnet_050',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif efficientnet_type == "mnasnet_075":
        try:
            efficientnet_model = create_model('mnasnet_075',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mnasnet_075',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif efficientnet_type == "mnasnet_100":
        try:
            efficientnet_model = create_model('mnasnet_100',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mnasnet_100',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif efficientnet_type == "mnasnet_140":
        try:
            efficientnet_model = create_model('mnasnet_140',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mnasnet_140',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif efficientnet_type == "semnasnet_050":
        try:
            efficientnet_model = create_model('semnasnet_050',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('semnasnet_050',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif efficientnet_type == "semnasnet_075":
        try:
            efficientnet_model = create_model('semnasnet_075',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('semnasnet_075',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif efficientnet_type == "semnasnet_100":
        try:
            efficientnet_model = create_model('semnasnet_100',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('semnasnet_100',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif efficientnet_type == "semnasnet_140":
        try:
            efficientnet_model = create_model('semnasnet_140',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('semnasnet_140',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif efficientnet_type == "mnasnet_small":
        try:
            efficientnet_model = create_model('mnasnet_small',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mnasnet_small',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif efficientnet_type == "mobilenetv2_035":
        try:
            efficientnet_model = create_model('mobilenetv2_035',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mobilenetv2_035',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif efficientnet_type == "mobilenetv2_050":
        try:
            efficientnet_model = create_model('mobilenetv2_050',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mobilenetv2_050',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif efficientnet_type == "mobilenetv2_075":
        try:
            efficientnet_model = create_model('mobilenetv2_075',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mobilenetv2_075',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif efficientnet_type == "mobilenetv2_100":
        try:
            efficientnet_model = create_model('mobilenetv2_100',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mobilenetv2_100',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif efficientnet_type == "mobilenetv2_140":
        try:
            efficientnet_model = create_model('mobilenetv2_140',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mobilenetv2_140',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif efficientnet_type == "mobilenetv2_110d":
        try:
            efficientnet_model = create_model('mobilenetv2_110d',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mobilenetv2_110d',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif efficientnet_type == "mobilenetv2_120d":
        try:
            efficientnet_model = create_model('mobilenetv2_120d',
                                         pretrained=True,
                                         num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mobilenetv2_120d',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif efficientnet_type == "fbnetc_100":
        try:
            efficientnet_model = create_model('fbnetc_100',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('fbnetc_100',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "spnasnet_100":
        try:
            efficientnet_model = create_model('spnasnet_100',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('spnasnet_100',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b0":
        try:
            efficientnet_model = create_model('efficientnet_b0',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b0',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b1":
        try:
            efficientnet_model = create_model('efficientnet_b1',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b1',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b2":
        try:
            efficientnet_model = create_model('efficientnet_b2',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b2',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b3":
        try:
            efficientnet_model = create_model('efficientnet_b3',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b3',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b4":
        try:
            efficientnet_model = create_model('efficientnet_b4',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b4',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b5":
        try:
            efficientnet_model = create_model('efficientnet_b5',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b5',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b6":
        try:
            efficientnet_model = create_model('efficientnet_b6',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b6',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b7":
        try:
            efficientnet_model = create_model('efficientnet_b7',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b7',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b8":
        try:
            efficientnet_model = create_model('efficientnet_b8',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b8',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_l2":
        try:
            efficientnet_model = create_model('efficientnet_l2',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_l2',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b0_gn":
        try:
            efficientnet_model = create_model('efficientnet_b0_gn',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b0_gn',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b0_g8_gn":
        try:
            efficientnet_model = create_model('efficientnet_b0_g8_gn',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b0_g8_gn',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b0_g16_evos":
        try:
            efficientnet_model = create_model('efficientnet_b0_g16_evos',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b0_g16_evos',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b3_gn":
        try:
            efficientnet_model = create_model('efficientnet_b3_gn',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b3_gn',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b3_g8_gn":
        try:
            efficientnet_model = create_model('efficientnet_b3_g8_gn',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b3_g8_gn',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_es":
        try:
            efficientnet_model = create_model('efficientnet_es',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_es',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_es_pruned":
        try:
            efficientnet_model = create_model('efficientnet_es_pruned',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_es_pruned',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_em":
        try:
            efficientnet_model = create_model('efficientnet_em',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_em',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_el":
        try:
            efficientnet_model = create_model('efficientnet_el',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_el',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_el_pruned":
        try:
            efficientnet_model = create_model('efficientnet_el_pruned',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_el_pruned',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_cc_b0_4e":
        try:
            efficientnet_model = create_model('efficientnet_cc_b0_4e',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_cc_b0_4e',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_cc_b0_8e":
        try:
            efficientnet_model = create_model('efficientnet_cc_b0_8e',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_cc_b0_8e',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_cc_b1_8e":
        try:
            efficientnet_model = create_model('efficientnet_cc_b1_8e',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_cc_b1_8e',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_lite0":
        try:
            efficientnet_model = create_model('efficientnet_lite0',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_lite0',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_lite1":
        try:
            efficientnet_model = create_model('efficientnet_lite1',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_lite1',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_lite2":
        try:
            efficientnet_model = create_model('efficientnet_lite2',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_lite2',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_lite3":
        try:
            efficientnet_model = create_model('efficientnet_lite3',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_lite3',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_lite4":
        try:
            efficientnet_model = create_model('efficientnet_lite4',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_lite4',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b1_pruned":
        try:
            efficientnet_model = create_model('efficientnet_b1_pruned',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b1_pruned',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b2_pruned":
        try:
            efficientnet_model = create_model('efficientnet_b2_pruned',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b2_pruned',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnet_b3_pruned":
        try:
            efficientnet_model = create_model('efficientnet_b3_pruned',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnet_b3_pruned',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnetv2_rw_t":
        try:
            efficientnet_model = create_model('efficientnetv2_rw_t',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnetv2_rw_t',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "gc_efficientnetv2_rw_t":
        try:
            efficientnet_model = create_model('gc_efficientnetv2_rw_t',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('gc_efficientnetv2_rw_t',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnetv2_rw_s":
        try:
            efficientnet_model = create_model('efficientnetv2_rw_s',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnetv2_rw_s',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnetv2_rw_m":
        try:
            efficientnet_model = create_model('efficientnetv2_rw_m',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnetv2_rw_m',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnetv2_s":
        try:
            efficientnet_model = create_model('efficientnetv2_s',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnetv2_s',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnetv2_m":
        try:
            efficientnet_model = create_model('efficientnetv2_m',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnetv2_m',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnetv2_l":
        try:
            efficientnet_model = create_model('efficientnetv2_l',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnetv2_l',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "efficientnetv2_xl":
        try:
            efficientnet_model = create_model('efficientnetv2_xl',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('efficientnetv2_xl',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_b0":
        try:
            efficientnet_model = create_model('tf_efficientnet_b0',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_b0',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_b1":
        try:
            efficientnet_model = create_model('tf_efficientnet_b1',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_b1',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_b2":
        try:
            efficientnet_model = create_model('tf_efficientnet_b2',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_b2',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_b3":
        try:
            efficientnet_model = create_model('tf_efficientnet_b3',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_b3',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_b4":
        try:
            efficientnet_model = create_model('tf_efficientnet_b4',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_b4',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_b5":
        try:
            efficientnet_model = create_model('tf_efficientnet_b5',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_b5',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_b6":
        try:
            efficientnet_model = create_model('tf_efficientnet_b6',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_b6',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_b7":
        try:
            efficientnet_model = create_model('tf_efficientnet_b7',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_b7',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_b8":
        try:
            efficientnet_model = create_model('tf_efficientnet_b8',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_b8',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_l2":
        try:
            efficientnet_model = create_model('tf_efficientnet_l2',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_l2',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_es":
        try:
            efficientnet_model = create_model('tf_efficientnet_es',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_es',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_em":
        try:
            efficientnet_model = create_model('tf_efficientnet_em',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_em',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_el":
        try:
            efficientnet_model = create_model('tf_efficientnet_el',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_el',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_cc_b0_4e":
        try:
            efficientnet_model = create_model('tf_efficientnet_cc_b0_4e',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_cc_b0_4e',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_cc_b0_8e":
        try:
            efficientnet_model = create_model('tf_efficientnet_cc_b0_8e',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_cc_b0_8e',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_cc_b1_8e":
        try:
            efficientnet_model = create_model('tf_efficientnet_cc_b1_8e',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_cc_b1_8e',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_lite0":
        try:
            efficientnet_model = create_model('tf_efficientnet_lite0',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_lite0',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_lite1":
        try:
            efficientnet_model = create_model('tf_efficientnet_lite1',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_lite1',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_lite2":
        try:
            efficientnet_model = create_model('tf_efficientnet_lite2',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_lite2',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_lite3":
        try:
            efficientnet_model = create_model('tf_efficientnet_lite3',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_lite3',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnet_lite4":
        try:
            efficientnet_model = create_model('tf_efficientnet_lite4',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnet_lite4',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnetv2_s":
        try:
            efficientnet_model = create_model('tf_efficientnetv2_s',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnetv2_s',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnetv2_m":
        try:
            efficientnet_model = create_model('tf_efficientnetv2_m',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnetv2_m',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnetv2_l":
        try:
            efficientnet_model = create_model('tf_efficientnetv2_l',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnetv2_l',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnetv2_xl":
        try:
            efficientnet_model = create_model('tf_efficientnetv2_xl',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnetv2_xl',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnetv2_b0":
        try:
            efficientnet_model = create_model('tf_efficientnetv2_b0',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnetv2_b0',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnetv2_b1":
        try:
            efficientnet_model = create_model('tf_efficientnetv2_b1',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnetv2_b1',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnetv2_b2":
        try:
            efficientnet_model = create_model('tf_efficientnetv2_b2',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnetv2_b2',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_efficientnetv2_b3":
        try:
            efficientnet_model = create_model('tf_efficientnetv2_b3',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_efficientnetv2_b3',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "mixnet_s":
        try:
            efficientnet_model = create_model('mixnet_s',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mixnet_s',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "mixnet_m":
        try:
            efficientnet_model = create_model('mixnet_m',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mixnet_m',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "mixnet_l":
        try:
            efficientnet_model = create_model('mixnet_l',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mixnet_l',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "mixnet_xl":
        try:
            efficientnet_model = create_model('mixnet_xl',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mixnet_xl',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "mixnet_xxl":
        try:
            efficientnet_model = create_model('mixnet_xxl',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('mixnet_xxl',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_mixnet_s":
        try:
            efficientnet_model = create_model('tf_mixnet_s',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_mixnet_s',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_mixnet_m":
        try:
            efficientnet_model = create_model('tf_mixnet_m',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_mixnet_m',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tf_mixnet_l":
        try:
            efficientnet_model = create_model('tf_mixnet_l',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tf_mixnet_l',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tinynet_a":
        try:
            efficientnet_model = create_model('tinynet_a',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tinynet_a',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tinynet_b":
        try:
            efficientnet_model = create_model('tinynet_b',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tinynet_b',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tinynet_c":
        try:
            efficientnet_model = create_model('tinynet_c',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tinynet_c',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tinynet_d":
        try:
            efficientnet_model = create_model('tinynet_d',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tinynet_d',
                                              pretrained=False,
                                              num_classes=num_classes)
    elif efficientnet_type == "tinynet_e":
        try:
            efficientnet_model = create_model('tinynet_e',
                                              pretrained=True,
                                              num_classes=num_classes)
        except ValueError:
            efficientnet_model = create_model('tinynet_e',
                                              pretrained=False,
                                              num_classes=num_classes)
    else:
        raise ValueError(f'Unknown EfficientNet Architecture: {efficientnet_type}')

    if torch_vision:
        # Modify last layer to suit number of classes
        num_features = efficientnet_model.classifier[-1].in_features
        efficientnet_model.classifier[-1] = nn.Linear(num_features, num_classes)

    return efficientnet_model
