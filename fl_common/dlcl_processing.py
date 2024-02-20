import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from captum.attr import (
    IntegratedGradients,
    GuidedBackprop,
)
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from fl_common.models.utils import (get_optimizer,
                                    get_criterion,
                                    get_scheduler,
                                    generate_xai_heatmaps,
                                    get_dataset,
                                    EarlyStopping)
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
from fl_common.models.densenet import get_densenet_model
from fl_common.models.edgenet import get_edgenet_model
from fl_common.models.efficientnet import get_efficientnet_model
from fl_common.models.fastvit import get_fastvit_model
from fl_common.models.gcvit import get_gcvit_model
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
from fl_common.models.vgg import get_vgg_model
from fl_common.models.vision_transformer import get_vision_model
from fl_common.models.volo import get_volo_model
from fl_common.models.vovnet import get_vovnet_model
from fl_common.models.wide_resnet import get_wide_resnet_model
from fl_common.models.xception import get_xception_model
from fl_common.models.xcit import get_xcit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# Dataset Parameters
dataset_path = 'c:/IA/Data'  # Replace with the actual path to the dataset
normalize_params = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
augmentation_params = {
    'data_augmentation': True,
    'random_rotation': 0.8,
    'rotation_range': 90,  # Augmentation plus importante de la rotation
    'horizontal_flip_prob': 0.8,  # Probabilité plus élevée de retournement horizontal
    'vertical_flip_prob': 0.8,  # Probabilité plus élevée de retournement vertical
    'resize': 224,
}
batch_size = 16

model_list = ['xcit_nano_12_p16_224', 'xception41', 'ResNet18', 'Swin_V2_T', 'RegNet_X_400MF', 'MobileNet_V3_Small',
              'ConvNeXt_Tiny', 'AlexNet', 'GoogLeNet', 'Inception_V3', 'VGG11', 'DenseNet121', 'EfficientNetB0',
              'ShuffleNet_V2_X0_5', 'MNASNet0_5', 'Wide_ResNet50_2', 'ResNeXt50_32X4D', 'MaxVit_T', 'SqueezeNet1_0',
              'ViT_B_16', 'volo_d1_224', 'edgenext_small', 'beit_base_patch16_224', 'convit_tiny', 'davit_tiny',
              'tiny_vit_5m_224', 'fastvit_t8', 'coat_tiny', 'cait_xxs24_224', 'gcvit_xxtiny', 'vovnet39a',
              'levit_128s', 'convmixer_1536_20', 'crossvit_tiny_240', 'cspresnet50']

# Model Parameters
best_val_loss = float('inf')  # Initialize the best validation loss

# Training Parameters
perform_second_training = True  # Set to True to perform the second training
perform_third_training = False  # Set To True to perform the third training
verbose = True

optimizer_name_phase1 = 'SGD'
learning_rate_phase1 = 0.01
criterion_name_phase1 = 'CrossEntropyLoss'
num_epochs_phase1 = 5  # Number of epochs for the first training phase
scheduler_phase1 = False
early_stopping_patience_phase1 = 5

optimizer_name_phase2 = 'SGD'
learning_rate_phase2 = 0.0005
criterion_name_phase2 = 'CrossEntropyLoss'
num_epochs_phase2 = 50  # Number of epochs for the second training phase
scheduler_phase2 = True
early_stopping_patience_phase2 = 5

optimizer_name_phase3 = 'SGD'
learning_rate_phase3 = 0.0001
criterion_name_phase3 = 'CrossEntropyLoss'
num_epochs_phase3 = 50  # Number of epochs for the third training phase
scheduler_phase3 = False
early_stopping_patience_phase3 = 5

xai = False


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
    Retrieves a model belonging to family D based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should be one of the supported models:
                        - For Davit models: ['davit_tiny', 'davit_small', 'davit_base', 'davit_large',
                                              'davit_huge', 'davit_giant']
                        - For DenseNet models: ['DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201']
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model if available, otherwise 'Unknown'.
    """
    model = "Unknown"

    if model_type in ['davit_tiny', 'davit_small', 'davit_base', 'davit_large', 'davit_huge', 'davit_giant']:
        model = get_davit_model(model_type, num_classes)
    elif model_type in ['DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201']:
        model = get_densenet_model(model_type, num_classes)

    return model


def get_family_model_e(model_type, num_classes):
    """
    Retrieves a model belonging to family E based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should be one of the supported models:
                        - For EdgeNet models: ['edgenext_xx_small', 'edgenext_x_small', 'edgenext_small',
                                               'edgenext_base', 'edgenext_small_rw']
                        - For EfficientNet models: ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2',
                                                    'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
                                                    'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2S',
                                                    'EfficientNetV2M', 'EfficientNetV2L']
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model if available, otherwise 'Unknown'.
    """
    model = "Unknown"

    if model_type in ['edgenext_xx_small', 'edgenext_x_small', 'edgenext_small', 'edgenext_base',
                      'edgenext_small_rw']:
        model = get_edgenet_model(model_type, num_classes)
    elif model_type in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4',
                        'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2S', 'EfficientNetV2M',
                        'EfficientNetV2L']:
        model = get_efficientnet_model(model_type, num_classes)

    return model


def get_family_model_f(model_type, num_classes):
    """
    Retrieves a model belonging to family F based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should be one of the supported models:
                        - For FastViT models: ['fastvit_t8', 'fastvit_t12', 'fastvit_s12', 'fastvit_sa12',
                                               'fastvit_sa24', 'fastvit_sa36', 'fastvit_ma36']
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model if available, otherwise 'Unknown'.
    """
    model = "Unknown"

    if model_type in ['fastvit_t8', 'fastvit_t12', 'fastvit_s12', 'fastvit_sa12', 'fastvit_sa24', 'fastvit_sa36',
                      'fastvit_ma36']:
        model = get_fastvit_model(model_type, num_classes)

    return model


def get_family_model_g(model_type, num_classes):
    """
    Retrieves a model belonging to family G based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should be one of the supported models:
                        - For GCViT models: ['gcvit_xxtiny', 'gcvit_xtiny', 'gcvit_tiny', 'gcvit_small',
                                              'gcvit_base']
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model if available, otherwise 'Unknown'.
    """
    model = "Unknown"

    if model_type in ['gcvit_xxtiny', 'gcvit_xtiny', 'gcvit_tiny', 'gcvit_small', 'gcvit_base']:
        model = get_gcvit_model(model_type, num_classes)

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
    Retrieves a model belonging to family T based on the provided model type and number of classes.

    Parameters:
    - model_type (str): The type of model to retrieve. Should be one of the supported models:
                        - For TinyViT models: ['tiny_vit_5m_224', 'tiny_vit_11m_224', 'tiny_vit_21m_224',
                                               'tiny_vit_21m_384', 'tiny_vit_21m_512']
    - num_classes (int): The number of classes for the model.

    Returns:
    - model: The requested model if available, otherwise 'Unknown'.
    """
    model = "Unknown"

    if model_type in ['tiny_vit_5m_224', 'tiny_vit_11m_224', 'tiny_vit_21m_224', 'tiny_vit_21m_384', 'tiny_vit_21m_512']:
        model = get_tiny_vit_model(model_type, num_classes)

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


for model_type in model_list:
    save_dir = 'c:/TFE/Models/' + model_type + '/'  # Replace with the actual path where to save results
    os.makedirs(save_dir, exist_ok=True)

    # Load your custom dataset
    train_loader, val_loader, test_loader, num_classes, class_names = get_dataset(dataset_path,
                                                                                  batch_size,
                                                                                  augmentation_params,
                                                                                  normalize_params)

    print(f"Model: {model_type}")

    # Use the pre-trained model
    model = get_family_model(model_type, num_classes)

    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    # Define the loss criterion and optimizer
    model_parameters = model.parameters()
    criterion = get_criterion(criterion_name_phase1)
    optimizer = get_optimizer(optimizer_name_phase1, model_parameters, learning_rate_phase1)

    # scheduler = get_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.5)
    # scheduler = get_scheduler(optimizer, scheduler_type='multi_step', milestones=[30, 60, 90], gamma=0.5)
    # scheduler = get_scheduler(optimizer, scheduler_type='exponential', gamma=0.95)

    scheduler = get_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.5)

    # Training loop
    early_stopping_phase1 = EarlyStopping(patience=early_stopping_patience_phase1, verbose=verbose)
    early_stopping_phase2 = EarlyStopping(patience=early_stopping_patience_phase2, verbose=verbose)
    early_stopping_phase3 = EarlyStopping(patience=early_stopping_patience_phase3, verbose=verbose)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    elapsed_times = []

    start_time = time.time()

    for epoch in range(num_epochs_phase1):
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs_phase1}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Use tqdm for a progress bar over the training batches
        with tqdm(enumerate(train_loader), total=len(train_loader),
                  desc=f"Epoch {epoch + 1}/{num_epochs_phase1}") as progress_bar:
            for batch_idx, (inputs, labels) in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                # Print progress within the epoch
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                    avg_batch_loss = running_loss / (batch_idx + 1)
                    batch_accuracy = correct_train / total_train
                    progress_bar.set_postfix(Batch_Loss=f"{avg_batch_loss:.4f}", Batch_Accuracy=f"{batch_accuracy:.4f}")

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(test_loader)
            val_losses.append(avg_val_loss)
            val_accuracy = correct_val / total_val
            val_accuracies.append(val_accuracy)

        # Update learning rate after the optimizer step
        if scheduler_phase1:
            scheduler.step()

        # Early stopping check
        early_stopping_phase1(avg_val_loss, model)
        if early_stopping_phase1.early_stop:
            print("Early stopping after {} epochs".format(epoch + 1))
            break

        # Save the model if the current validation loss is the best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_dir + 'best_model.pth')

        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - epoch_start_time
        elapsed_times.append(elapsed_time)

        # Print and plot results
        elapsed_time_msg = "Elapsed Time: "

        if elapsed_time >= 3600:
            elapsed_time_msg += f"{elapsed_time / 3600:.2f} hour(s)"
        elif elapsed_time >= 60:
            elapsed_time_msg += f"{elapsed_time / 60:.2f} minute(s)"
        else:
            elapsed_time_msg += f"{elapsed_time:.2f} seconds"

        print(f"Epoch {epoch + 1}/{num_epochs_phase1} => "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
              f"{elapsed_time_msg}")

    if perform_second_training and not early_stopping_phase1.early_stop:  # Proceed only if the first phase didn't early stop
        print("\nStarting the second training phase...\n")

        # Optionally reset the optimizer, criterion and scheduler for the second phase
        model_parameters = model.parameters()
        criterion = get_criterion(criterion_name_phase2)
        optimizer = get_optimizer(optimizer_name_phase2, model_parameters, learning_rate_phase2)

        # scheduler = get_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.5)
        # scheduler = get_scheduler(optimizer, scheduler_type='multi_step', milestones=[30, 60, 90], gamma=0.5)
        # scheduler = get_scheduler(optimizer, scheduler_type='exponential', gamma=0.95)

        scheduler = get_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.5)

        for epoch in range(num_epochs_phase2):
            epoch_start_time = time.time()

            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            print(f"\nEpoch {epoch + 1}/{num_epochs_phase2}, Learning Rate: {optimizer.param_groups[0]['lr']}")

            # Use tqdm for a progress bar over the training batches
            with tqdm(enumerate(train_loader), total=len(train_loader),
                      desc=f"Epoch {epoch + 1}/{num_epochs_phase2}") as progress_bar:
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()

                    # Print progress within the epoch
                    if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                        avg_batch_loss = running_loss / (batch_idx + 1)
                        batch_accuracy = correct_train / total_train
                        progress_bar.set_postfix(Batch_Loss=f"{avg_batch_loss:.4f}",
                                                 Batch_Accuracy=f"{batch_accuracy:.4f}")

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)

            # Validation
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

                avg_val_loss = val_loss / len(test_loader)
                val_losses.append(avg_val_loss)
                val_accuracy = correct_val / total_val
                val_accuracies.append(val_accuracy)

            # Update learning rate after the optimizer step
            if scheduler_phase2:
                scheduler.step()

            # Early stopping check for the second phase
            early_stopping_phase2(avg_val_loss, model)
            if early_stopping_phase2.early_stop:
                print("Early stopping after the second training phase.")
                break

            # Save the model if the current validation loss is the best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_dir + 'best_model.pth')

            epoch_end_time = time.time()
            elapsed_time = epoch_end_time - epoch_start_time
            elapsed_times.append(elapsed_time)

            # Print and plot results for the second phase
            elapsed_time_msg = "Elapsed Time: "

            if elapsed_time >= 3600:
                elapsed_time_msg += f"{elapsed_time / 3600:.2f} hour(s)"
            elif elapsed_time >= 60:
                elapsed_time_msg += f"{elapsed_time / 60:.2f} minute(s)"
            else:
                elapsed_time_msg += f"{elapsed_time:.2f} seconds"

            print(f"\nEpoch {epoch + 1}/{num_epochs_phase2} (Phase 2) => "
                  f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
                  f"{elapsed_time_msg}")

    # Third Training Phase (Optional)
    if perform_third_training and not early_stopping_phase2.early_stop:  # Proceed only if the second phase didn't early stop
        print("\nStarting the third training phase...\n")

        # Optionally reset the optimizer, criterion and scheduler for the second phase
        model_parameters = model.parameters()
        criterion = get_criterion(criterion_name_phase3)
        optimizer = get_optimizer(optimizer_name_phase3, model_parameters, learning_rate_phase3)

        # scheduler = get_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.5)
        # scheduler = get_scheduler(optimizer, scheduler_type='multi_step', milestones=[30, 60, 90], gamma=0.5)
        # scheduler = get_scheduler(optimizer, scheduler_type='exponential', gamma=0.95)

        scheduler = get_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.5)

        for epoch in range(num_epochs_phase3):
            epoch_start_time = time.time()

            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            print(f"\nEpoch {epoch + 1}/{num_epochs_phase3}, Learning Rate: {optimizer.param_groups[0]['lr']}")

            # Use tqdm for a progress bar over the training batches
            with tqdm(enumerate(train_loader), total=len(train_loader),
                      desc=f"Epoch {epoch + 1}/{num_epochs_phase3}") as progress_bar:
                for batch_idx, (inputs, labels) in progress_bar:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()

                    # Print progress within the epoch
                    if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                        avg_batch_loss = running_loss / (batch_idx + 1)
                        batch_accuracy = correct_train / total_train
                        progress_bar.set_postfix(Batch_Loss=f"{avg_batch_loss:.4f}",
                                                 Batch_Accuracy=f"{batch_accuracy:.4f}")

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)

            # Validation
            model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

                avg_val_loss = val_loss / len(test_loader)
                val_losses.append(avg_val_loss)
                val_accuracy = correct_val / total_val
                val_accuracies.append(val_accuracy)

            # Update learning rate after the optimizer step
            if scheduler_phase3:
                scheduler.step()

            # Early stopping check for the third phase
            early_stopping_phase3(avg_val_loss, model)
            if early_stopping_phase3.early_stop:
                print("Early stopping after the third training phase.")
                break

            # Save the model if the current validation loss is the best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_dir + 'best_model.pth')

            epoch_end_time = time.time()
            elapsed_time = epoch_end_time - epoch_start_time
            elapsed_times.append(elapsed_time)

            # Print and plot results for the third phase
            elapsed_time_msg = "Elapsed Time: "

            if elapsed_time >= 3600:
                elapsed_time_msg += f"{elapsed_time / 3600:.2f} hour(s)"
            elif elapsed_time >= 60:
                elapsed_time_msg += f"{elapsed_time / 60:.2f} minute(s)"
            else:
                elapsed_time_msg += f"{elapsed_time:.2f} seconds"

            print(f"\nEpoch {epoch + 1}/{num_epochs_phase3} (Phase 3) => "
                  f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
                  f"{elapsed_time_msg}")

    # Calculate total training time
    total_training_time = time.time() - start_time
    if total_training_time >= 3600:
        unit, time_value = 'hour(s)', total_training_time / 3600
    elif total_training_time >= 60:
        unit, time_value = 'minute(s)', total_training_time / 60
    else:
        unit, time_value = 'seconds', total_training_time

    print(f"\nTotal Training Time: {time_value:.2f} {unit}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot losses
    axes[0].plot(train_losses, label='Training Loss')
    axes[0].plot(val_losses, label='Validation Loss')
    axes[0].set_title('Training and Validation Losses')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot accuracies
    axes[1].plot(train_accuracies, label='Training Accuracy')
    axes[1].plot(val_accuracies, label='Validation Accuracy')
    axes[1].set_title('Training and Validation Accuracies')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    fig.tight_layout()

    # Saving the graph
    save_path = os.path.join(save_dir, 'training_curves.png')
    fig.savefig(save_path)

    plt.show()

    # Test the final model on the test set
    model.eval()
    correct_test = 0
    total_test = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print final results
    test_accuracy = correct_test / total_test
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    # Generate and plot confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=plt.cm.Blues, cbar=False, annot_kws={"size": 14})
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_dir + 'confusion_matrix.png')  # Saving the confusion matrix
    plt.show()  # Confusion matrix display

    # Print classification report
    class_report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:\n", class_report)

    # Save classification report to a text file
    with open(save_dir + 'classification_report.txt', 'w') as report_file:
        report_file.write(save_dir + "Classification Report:\n" + class_report)

    # Loop through test dataset and generate XAI heatmaps for specific methods
    if xai:
        save_dir += 'xai_heatmaps/'
        # Loop through test dataset and generate XAI heatmaps for specific methods
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Convert predicted and labels to scalar values
            predicted_scalars = predicted.tolist()  # Convert to list
            labels_scalars = labels.tolist()  # Convert to list

            for j, (predicted_scalar, label_scalar) in enumerate(zip(predicted_scalars, labels_scalars)):
                if predicted_scalar != label_scalar:
                    print(
                        f"Example {i * test_loader.batch_size + j + 1}: Prediction: {predicted_scalar}, Actual: {label_scalar}")

                    # Specify the methods you want to use (e.g., 'GuidedBackprop' and 'IntegratedGradients')
                    specific_methods = [GuidedBackprop(model), IntegratedGradients(model)]

                    # Create a directory for XAI heatmaps based on the specific example
                    example_dir = f"{save_dir}/example_{i * test_loader.batch_size + j + 1}/"
                    os.makedirs(example_dir, exist_ok=True)

                    generate_xai_heatmaps(model, inputs[j], label_scalar, save_dir=example_dir,
                                          methods=specific_methods)
