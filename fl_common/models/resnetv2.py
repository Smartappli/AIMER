from timm import create_model


def get_resnetv2_model(resnetv2_type, num_classes):
    """
    Load a ResNetV2 model based on the specified type.

    Parameters:
        resnetv2_type (str): The type of ResNetV2 model to load.
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The loaded ResNetV2 model.

    Raises:
        ValueError: If the specified ResNetV2 architecture is unknown.
    """
    valid_types = {
        "resnetv2_50x1_bit",
        "resnetv2_50x3_bit",
        "resnetv2_101x1_bit",
        "resnetv2_101x3_bit",
        "resnetv2_152x2_bit",
        "resnetv2_152x4_bit",
        "resnetv2_50",
        "resnetv2_50d",
        "resnetv2_50t",
        "resnetv2_101",
        "resnetv2_101d",
        "resnetv2_152",
        "resnetv2_152d",
        "resnetv2_50d_gn",
        "resnetv2_50d_evos",
        "resnetv2_50d_frn",
    }

    if resnetv2_type not in valid_types:
        msg = f"Unknown ResNet v2 Architecture: {resnetv2_type}"
        raise ValueError(msg)

    try:
        return create_model(
            resnetv2_type,
            pretrained=True,
            num_classes=num_classes,
        )
    except RuntimeError as e:
        print(f"Error loading pretrained model: {e}")
        return create_model(
            resnetv2_type,
            pretrained=False,
            num_classes=num_classes,
        )
