from timm import create_model


def get_ghostnet_model(ghostnet_type, num_classes):
    """
    Get a GhostNet model based on the specified architecture type.

    Parameters:
        ghostnet_type (str): Type of GhostNet architecture.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: The selected GhostNet model instance.

    Raises:
        ValueError: If an unknown GhostNet architecture type is specified.
    """
    supported_types = {
        "ghostnet_050",
        "ghostnet_100",
        "ghostnet_130",
        "ghostnetv2_100",
        "ghostnetv2_130",
        "ghostnetv2_160",
    }

    if ghostnet_type not in supported_types:
        msg = f"Unknown GhostNet Architecture: {ghostnet_type}"
        raise ValueError(msg)

    try:
        return create_model(
            ghostnet_type, pretrained=True, num_classes=num_classes,
        )
    except RuntimeError as e:
        print(f"{ghostnet_type} - Error loading pretrained model: {e}")
        return create_model(
            ghostnet_type, pretrained=False, num_classes=num_classes,
        )
