from torch import nn
from torchvision import models
from timm import create_model


def get_convnext_model(convnext_type, num_classes):
    """
    Obtain a ConvNeXt model with a specified architecture type and modify it for the given number of classes.

    Args:
    - convnext_type (str): Type of ConvNeXt architecture to be loaded. Options: 'ConvNeXt_Tiny', 'ConvNeXt_Small',
      'ConvNeXt_Base', 'ConvNeXt_Large'. Default is 'ConvNeXt_Large'.
    - num_classes (int): Number of output classes for the modified model. Default is 1000.

    Returns:
    - convnext_model (torch.nn.Module): The modified ConvNeXt model.

    Raises:
    - ValueError: If the provided convnext_type is not recognized or the model structure is not as expected.
    """
    # Mapping of vision types to their corresponding torchvision models and
    # weights
    torchvision_models = {
        "ConvNeXt_Tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights),
        "ConvNeXt_Small": (models.convnext_small, models.ConvNeXt_Small_Weights),
        "ConvNeXt_Base": (models.convnext_base, models.ConvNeXt_Base_Weights),
        "ConvNeXt_Large": (models.convnext_large, models.ConvNeXt_Large_Weights),
    }

    timm_models = [
        "convnext_atto",
        "convnext_atto_ols",
        "convnext_femto",
        "convnext_femto_ols",
        "convnext_pico",
        "convnext_pico_ols",
        "convnext_nano",
        "convnext_nano_ols",
        "convnext_tiny_hnf",
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
        "convnext_large_mlp",
        "convnext_xlarge",
        "convnext_xxlarge",
        "convnextv2_atto",
        "convnextv2_femto",
        "convnextv2_pico",
        "convnextv2_nano",
        "convnextv2_small",
        "convnextv2_tiny",
        "convnextv2_base",
        "convnextv2_large",
        "convnextv2_huge",
    ]

    # Check if the vision type is from torchvision
    if convnext_type in torchvision_models:
        model_func, weights_class = torchvision_models[convnext_type]
        try:
            weights = weights_class.DEFAULT
            convnext_model = model_func(weights=weights)
        except RuntimeError as e:
            print(f"{convnext_type} - Error loading pretrained model: {e}")
            convnext_model = model_func(weights=None)

        # Modify last layer to suit number of classes
        for layer in reversed(convnext_model.classifier):
            if isinstance(layer, nn.Linear):
                # num_features = layer.in_features
                layer.out_features = num_classes
                break

    # Check if the vision type is from the 'timm' library
    elif convnext_type in timm_models:
        try:
            convnext_model = create_model(
                convnext_type, pretrained=True, num_classes=num_classes
            )
        except RuntimeError as e:
            print(f"{convnext_type} - Error loading pretrained model: {e}")
            convnext_model = create_model(
                convnext_type, pretrained=False, num_classes=num_classes
            )
    else:
        raise ValueError(f"Unknown ConvNeXt Architecture : {convnext_type}")

    return convnext_model
