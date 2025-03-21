from timm import create_model
from torch import nn
from torchvision import models


def get_inception_model(inception_type, num_classes):
    """
    Load a pre-trained Inception model of the specified type and modify its
    last layer to accommodate the given number of classes.

    Parameters:
    - inception_type (str): Type of Inception architecture, supported types:
        - 'Inception_V3'
        - 'inception_v4'
        - 'inception_resnet_v2'
    - num_classes (int): Number of output classes for the modified last layer.

    Returns:
    - torch.nn.Module: Modified Inception model with the specified architecture
      and last layer adapted for the given number of classes.

    Raises:
    - ValueError: If an unknown Inception architecture type is provided.
    """
    # Mapping of vision types to their corresponding torchvision models and
    # weights
    torchvision_models = {
        "Inception_V3": (models.inception_v3, models.Inception_V3_Weights),
    }

    timm_models = ["inception_v4", "inception_resnet_v2"]

    # Check if the vision type is from torchvision
    if inception_type in torchvision_models:
        model_func, weights_class = torchvision_models[inception_type]
        try:
            weights = weights_class.DEFAULT
            inception_model = model_func(weights=weights)
        except RuntimeError as e:
            print(f"{inception_type} - Error loading pretrained model: {e}")
            inception_model = model_func(weights=None)

        # Modify the last layer to suit the given number of classes
        num_features = inception_model.fc.in_features
        inception_model.fc = nn.Linear(num_features, num_classes)

    # Check if the vision type is from the 'timm' library
    elif inception_type in timm_models:
        try:
            inception_model = create_model(
                inception_type,
                pretrained=True,
                num_classes=num_classes,
            )
        except RuntimeError as e:
            print(f"{inception_type} - Error loading pretrained model: {e}")
            inception_model = create_model(
                inception_type,
                pretrained=False,
                num_classes=num_classes,
            )
    else:
        msg = f"Unknown Inception Architecture: {inception_type}"
        raise ValueError(msg)

    return inception_model
