from torch import nn
from torchvision import models


def get_mnasnet_model(mnasnet_type: str, num_classes: int) -> nn.Module:
    """
    Load a pre-trained MNASNet model of the specified type and modify its
    last layer to accommodate the given number of classes.
    """
    # Mapping of MNASNet types to their corresponding weight classes
    weight_classes = {
        "MNASNet0_5": models.MNASNet0_5_Weights,
        "MNASNet0_75": models.MNASNet0_75_Weights,
        "MNASNet1_0": models.MNASNet1_0_Weights,
        "MNASNet1_3": models.MNASNet1_3_Weights,
    }

    # Check if the specified type is valid
    if mnasnet_type not in weight_classes:
        msg = f"Unknown MNASNet Architecture: {mnasnet_type}"
        raise ValueError(msg)

    # Load the pre-trained model
    try:
        weights = weight_classes[mnasnet_type].DEFAULT
        mnasnet_model = getattr(models.mnasnet, mnasnet_type.lower())(
            weights=weights,
        )
    except RuntimeError as e:
        print(f"{mnasnet_type} - Error loading pretrained model: {e}")
        mnasnet_model = getattr(models.mnasnet, mnasnet_type.lower())(
            weights=None,
        )

    # Modify the last layer
    num_features = mnasnet_model.classifier[1].in_features
    mnasnet_model.classifier[1] = nn.Linear(num_features, num_classes)

    return mnasnet_model
