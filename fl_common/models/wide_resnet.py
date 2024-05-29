from torch import nn
from torchvision import models


def get_wide_resnet_model(wide_resnet_type, num_classes):
    """
    Returns a modified Wide ResNet model based on the specified type.

    Parameters:
        - wide_resnet_type (str): Type of Wide ResNet architecture.
                                 Currently supports 'Wide_ResNet50_2' and 'Wide_ResNet101_2'.
        - num_classes (int): Number of classes for the modified last layer.

    Returns:
        - torch.nn.Module: Modified Wide ResNet model with the specified number of classes.

    Raises:
        - ValueError: If an unknown Wide ResNet architecture is provided.
    """
    # Dictionary mapping model types to their corresponding functions and
    # default weights
    wide_resnet_models = {
        "Wide_ResNet50_2": (
            models.wide_resnet50_2,
            models.Wide_ResNet50_2_Weights,
        ),
        "Wide_ResNet101_2": (
            models.wide_resnet101_2,
            models.Wide_ResNet101_2_Weights,
        ),
    }

    # Get the model function and default weights based on the specified type
    if wide_resnet_type in wide_resnet_models:
        model_func, weights_class = wide_resnet_models[wide_resnet_type]
        try:
            weights = weights_class.DEFAULT
            wide_resnet_model = model_func(weights=weights)
        except RuntimeError as e:
            print(f"{wide_resnet_type} - Error loading pretrained model: {e}")
            wide_resnet_model = model_func(weights=None)
    else:
        msg = f"Unknown Wide ResNet Architecture: {wide_resnet_type}"
        raise ValueError(msg)

    # Modify the last layer to suit the given number of classes
    num_features = wide_resnet_model.fc.in_features
    wide_resnet_model.fc = nn.Linear(num_features, num_classes)

    return wide_resnet_model
