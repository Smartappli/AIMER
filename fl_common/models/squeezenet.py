import torch.nn as nn
from torchvision import models


def get_squeezenet_model(squeezenet_type, num_classes):
    """
    Returns a modified SqueezeNet model based on the specified type.
    """

    # Mapping of SqueezeNet types to their respective functions and weights
    squeezenet_versions = {
        "SqueezeNet1_0": (models.squeezenet1_0, models.SqueezeNet1_0_Weights),
        "SqueezeNet1_1": (models.squeezenet1_1, models.SqueezeNet1_1_Weights),
    }

    # Validate squeezenet_type
    if squeezenet_type not in squeezenet_versions:
        raise ValueError(f"Unknown SqueezeNet Architecture: {squeezenet_type}")

    # Load the pre-trained model
    model_func, weights = squeezenet_versions[squeezenet_type]
    try:
        squeezenet_model = model_func(weights=weights.DEFAULT)
    except RuntimeError as e:
        print(f"{squeezenet_type} - Error loading pretrained model: {e}")
        squeezenet_model = model_func(weights=None)

    # Modify the last layer to suit the given number of classes
    num_features = squeezenet_model.classifier[1].in_channels
    squeezenet_model.classifier[1] = nn.Conv2d(
        num_features, num_classes, kernel_size=(1, 1), stride=(1, 1)
    )

    return squeezenet_model
