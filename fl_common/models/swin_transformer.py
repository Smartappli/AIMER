from timm import create_model
from torch import nn
from torchvision import models


def get_swin_transformer_model(swin_type, num_classes):
    """
    Loads a pre-trained Swin Transformer model based on the specified Swin type
    and modifies the last layer for a given number of output classes.

    Parameters:
    - swin_type (str, optional): Type of Swin Transformer model. Default is 'Swin_T'.
    - num_classes (int, optional): Number of output classes. Default is 1000.

    Returns:
    - swin_model (torch.nn.Module): Modified Swin Transformer model with the last layer
      adjusted for the specified number of output classes.

    Raises:
    - ValueError: If the specified Swin type is unknown or if the model does not have
      a known structure with a linear last layer.
    """
    # Mapping of vision types to their corresponding torchvision models and
    # weights
    torchvision_models = {
        "Swin_T": (models.swin_t, models.Swin_T_Weights),
        "Swin_S": (models.swin_s, models.Swin_S_Weights),
        "Swin_B": (models.swin_b, models.Swin_B_Weights),
        "Swin_V2_T": (models.swin_v2_t, models.Swin_V2_T_Weights),
        "Swin_V2_S": (models.swin_v2_s, models.Swin_V2_S_Weights),
        "Swin_V2_B": (models.swin_v2_b, models.Swin_V2_B_Weights),
    }

    timm_models = [
        "swin_tiny_patch4_window7_224",
        "swin_small_patch4_window7_224",
        "swin_base_patch4_window7_224",
        "swin_base_patch4_window12_384",
        "swin_large_patch4_window7_224",
        "swin_large_patch4_window12_384",
        "swin_s3_tiny_224",
        "swin_s3_small_224",
        "swin_s3_base_224",
    ]

    # Check if the vision type is from torchvision
    if swin_type in torchvision_models:
        model_func, weights_class = torchvision_models[swin_type]
        try:
            weights = weights_class.DEFAULT
            swin_model = model_func(weights=weights)
        except RuntimeError as e:
            print(f"{swin_type} - Error loading pretrained model: {e}")
            swin_model = model_func(weights=None)

        # Modify last layer to suit number of classes
        if hasattr(swin_model, "head") and isinstance(
            swin_model.head,
            nn.Linear,
        ):
            num_features = swin_model.head.in_features
            swin_model.head = nn.Linear(num_features, num_classes)
        else:
            msg = "Model does not have a known structure."
            raise ValueError(msg)

    # Check if the vision type is from the 'timm' library
    elif swin_type in timm_models:
        try:
            swin_model = create_model(
                swin_type,
                pretrained=True,
                num_classes=num_classes,
            )
        except RuntimeError as e:
            print(f"{swin_type} - Error loading pretrained model: {e}")
            swin_model = create_model(
                swin_type,
                pretrained=False,
                num_classes=num_classes,
            )
    else:
        msg = f"Unknown Swin Transformer Architecture: {swin_type}"
        raise ValueError(msg)

    return swin_model
