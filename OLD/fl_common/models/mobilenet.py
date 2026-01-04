from timm import create_model
from torch import nn
from torchvision import models


def get_mobilenet_model(mobilenet_type, num_classes):
    """
    Obtain a MobileNet model with a specified architecture type and modify it for the given number of classes.

    Args:
    - mobilenet_type (str): Type of MobileNet architecture to be loaded.
    - num_classes (int): Number of output classes for the modified model. Default is 1000.

    Returns:
    - mobilenet_model (torch.nn.Module): The modified MobileNet model.

    Raises:
    - ValueError: If the provided mobilenet_type is not recognized.

    Note:
    - This function loads a pre-trained MobileNet model and modifies its last fully connected layer
      to match the specified number of output classes.

    Example Usage:
    ```python
    # Obtain a MobileNet_V2 model with 10 output classes
    model = get_mobilenet_model(mobilenet_type='MobileNet_V2', num_classes=10)
    ```
    """
    # Mapping of vision types to their corresponding torchvision models and
    # weights
    torchvision_models = {
        "MobileNet_V2": (models.mobilenet_v2, models.MobileNet_V2_Weights),
        "MobileNet_V3_Small": (
            models.mobilenet_v3_small,
            models.MobileNet_V3_Small_Weights,
        ),
        "MobileNet_V3_Large": (
            models.mobilenet_v3_large,
            models.MobileNet_V3_Large_Weights,
        ),
    }

    timm_models = [
        "mobilenetv3_large_075",
        "mobilenetv3_large_100",
        "mobilenetv3_large_150d",
        "mobilenetv3_small_050",
        "mobilenetv3_small_075",
        "mobilenetv3_small_100",
        "mobilenetv3_rw",
        "tf_mobilenetv3_large_075",
        "tf_mobilenetv3_large_100",
        "tf_mobilenetv3_large_minimal_100",
        "tf_mobilenetv3_small_075",
        "tf_mobilenetv3_small_100",
        "tf_mobilenetv3_small_minimal_100",
        "fbnetv3_b",
        "fbnetv3_d",
        "fbnetv3_g",
        "lcnet_035",
        "lcnet_050",
        "lcnet_075",
        "lcnet_100",
        "lcnet_150",
        "mobilenetv4_conv_small_035",
        "mobilenetv4_conv_small_050",
        "mobilenetv4_conv_small",
        "mobilenetv4_conv_medium",
        "mobilenetv4_conv_large",
        "mobilenetv4_hybrid_medium",
        "mobilenetv4_hybrid_large",
        "mobilenetv4_conv_aa_medium",
        "mobilenetv4_conv_blur_medium",
        "mobilenetv4_conv_aa_large",
        "mobilenetv4_hybrid_medium_075",
        "mobilenetv4_hybrid_large_075",
    ]

    # Check if the vision type is from torchvision
    if mobilenet_type in torchvision_models:
        model_func, weights_class = torchvision_models[mobilenet_type]
        try:
            weights = weights_class.DEFAULT
            mobilenet_model = model_func(weights=weights)
        except RuntimeError as e:
            print(f"{mobilenet_type} - Error loading pretrained model: {e}")
            mobilenet_model = model_func(weights=None)

        # Modify last layer to suit number of classes
        num_features = mobilenet_model.classifier[-1].in_features
        mobilenet_model.classifier[-1] = nn.Linear(num_features, num_classes)

    # Check if the vision type is from the 'timm' library
    elif mobilenet_type in timm_models:
        try:
            mobilenet_model = create_model(
                mobilenet_type,
                pretrained=True,
                num_classes=num_classes,
            )
        except RuntimeError as e:
            print(f"{mobilenet_type} - Error loading pretrained model: {e}")
            mobilenet_model = create_model(
                mobilenet_type,
                pretrained=False,
                num_classes=num_classes,
            )
    else:
        msg = f"Unknown MobileNet Architecture : {mobilenet_type}"
        raise ValueError(msg)

    return mobilenet_model
