from timm import create_model
from torch import nn
from torchvision import models


def get_efficientnet_model(efficientnet_type, num_classes):
    """
    Obtain an EfficientNet model with a specified architecture type and modify it for the given number of classes.

    Args:
    - efficientnet_type (str): Type of EfficientNet architecture to be loaded.
    - num_classes (int): Number of output classes for the modified model. Default is 1000.

    Returns:
    - efficientnet_model (torch.nn.Module): The modified EfficientNet model.

    Raises:
    - ValueError: If the provided efficientnet_type is not recognized.
    """
    # Mapping of vision types to their corresponding torchvision models and
    # weights
    torchvision_models = {
        "EfficientNetB0": (
            models.efficientnet_b0,
            models.EfficientNet_B0_Weights,
        ),
        "EfficientNetB1": (
            models.efficientnet_b1,
            models.EfficientNet_B1_Weights,
        ),
        "EfficientNetB2": (
            models.efficientnet_b2,
            models.EfficientNet_B2_Weights,
        ),
        "EfficientNetB3": (
            models.efficientnet_b3,
            models.EfficientNet_B3_Weights,
        ),
        "EfficientNetB4": (
            models.efficientnet_b4,
            models.EfficientNet_B4_Weights,
        ),
        "EfficientNetB5": (
            models.efficientnet_b5,
            models.EfficientNet_B5_Weights,
        ),
        "EfficientNetB6": (
            models.efficientnet_b6,
            models.EfficientNet_B6_Weights,
        ),
        "EfficientNetB7": (
            models.efficientnet_b7,
            models.EfficientNet_B7_Weights,
        ),
        "EfficientNetV2S": (
            models.efficientnet_v2_s,
            models.EfficientNet_V2_S_Weights,
        ),
        "EfficientNetV2M": (
            models.efficientnet_v2_m,
            models.EfficientNet_V2_M_Weights,
        ),
        "EfficientNetV2L": (
            models.efficientnet_v2_l,
            models.EfficientNet_V2_L_Weights,
        ),
    }

    timm_models = [
        "mnasnet_050",
        "mnasnet_075",
        "mnasnet_100",
        "mnasnet_140",
        "semnasnet_050",
        "semnasnet_075",
        "semnasnet_100",
        "semnasnet_140",
        "mnasnet_small",
        "mobilenetv1_100",
        "mobilenetv1_100h",
        "mobilenetv1_125",
        "mobilenetv2_035",
        "mobilenetv2_050",
        "mobilenetv2_075",
        "mobilenetv2_100",
        "mobilenetv2_140",
        "mobilenetv2_110d",
        "mobilenetv2_120d",
        "fbnetc_100",
        "spnasnet_100",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnet_b5",
        "efficientnet_b6",
        "efficientnet_b7",
        "efficientnet_b8",
        "efficientnet_l2",
        "efficientnet_b0_gn",
        "efficientnet_b0_g8_gn",
        "efficientnet_b0_g16_evos",
        "efficientnet_b3_gn",
        "efficientnet_b3_g8_gn",
        "efficientnet_blur_b0",
        "efficientnet_es",
        "efficientnet_es_pruned",
        "efficientnet_em",
        "efficientnet_el",
        "efficientnet_el_pruned",
        "efficientnet_cc_b0_4e",
        "efficientnet_cc_b0_8e",
        "efficientnet_cc_b1_8e",
        "efficientnet_lite0",
        "efficientnet_lite1",
        "efficientnet_lite2",
        "efficientnet_lite3",
        "efficientnet_lite4",
        "efficientnet_b1_pruned",
        "efficientnet_b2_pruned",
        "efficientnet_b3_pruned",
        "efficientnetv2_rw_t",
        "gc_efficientnetv2_rw_t",
        "efficientnetv2_rw_s",
        "efficientnetv2_rw_m",
        "efficientnetv2_s",
        "efficientnetv2_m",
        "efficientnetv2_l",
        "efficientnetv2_xl",
        "tf_efficientnet_b0",
        "tf_efficientnet_b1",
        "tf_efficientnet_b2",
        "tf_efficientnet_b3",
        "tf_efficientnet_b4",
        "tf_efficientnet_b5",
        "tf_efficientnet_b6",
        "tf_efficientnet_b7",
        "tf_efficientnet_b8",
        "tf_efficientnet_l2",
        "tf_efficientnet_es",
        "tf_efficientnet_em",
        "tf_efficientnet_el",
        "tf_efficientnet_cc_b0_4e",
        "tf_efficientnet_cc_b0_8e",
        "tf_efficientnet_cc_b1_8e",
        "tf_efficientnet_lite0",
        "tf_efficientnet_lite1",
        "tf_efficientnet_lite2",
        "tf_efficientnet_lite3",
        "tf_efficientnet_lite4",
        "tf_efficientnetv2_s",
        "tf_efficientnetv2_m",
        "tf_efficientnetv2_l",
        "tf_efficientnetv2_xl",
        "tf_efficientnetv2_b0",
        "tf_efficientnetv2_b1",
        "tf_efficientnetv2_b2",
        "tf_efficientnetv2_b3",
        "efficientnet_x_b3",
        "efficientnet_x_b5",
        "efficientnet_h_b5",
        "mixnet_s",
        "mixnet_m",
        "mixnet_l",
        "mixnet_xl",
        "mixnet_xxl",
        "tf_mixnet_s",
        "tf_mixnet_m",
        "tf_mixnet_l",
        "tinynet_a",
        "tinynet_b",
        "tinynet_c",
        "tinynet_d",
        "tinynet_e",
        "mobilenet_edgetpu_100",
        "mobilenet_edgetpu_v2_xs",
        "mobilenet_edgetpu_v2_s",
        "mobilenet_edgetpu_v2_m",
        "mobilenet_edgetpu_v2_l",
    ]

    # Check if the vision type is from torchvision
    if efficientnet_type in torchvision_models:
        model_func, weights_class = torchvision_models[efficientnet_type]
        try:
            weights = weights_class.DEFAULT
            efficientnet_model = model_func(weights=weights)
        except RuntimeError as e:
            print(f"{efficientnet_type} - Error loading pretrained model: {e}")
            efficientnet_model = model_func(weights=None)

        # Modify last layer to suit number of classes
        num_features = efficientnet_model.classifier[-1].in_features
        efficientnet_model.classifier[-1] = nn.Linear(num_features, num_classes)

    # Check if the vision type is from the 'timm' library
    elif efficientnet_type in timm_models:
        try:
            efficientnet_model = create_model(
                efficientnet_type,
                pretrained=True,
                num_classes=num_classes,
            )
        except RuntimeError as e:
            print(f"{efficientnet_type} - Error loading pretrained model: {e}")
            efficientnet_model = create_model(
                efficientnet_type,
                pretrained=False,
                num_classes=num_classes,
            )
    else:
        msg = f"Unknown EfficientNet Architecture: {efficientnet_type}"
        raise ValueError(msg)

    return efficientnet_model
