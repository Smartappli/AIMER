from timm import create_model


def get_cspnet_model(cspnet_type, num_classes):
    """
    Creates and returns a CSPNet model based on the specified architecture type.

    Parameters:
        cspnet_type (str): The type of CSPNet architecture to use.
        num_classes (int): The number of classes for the final classification layer.

    Returns:
        torch.nn.Module: The CSPNet model with the specified architecture and number of classes.

    Raises:
        ValueError: If the specified CSPNet architecture type is unknown.
    """
    cspnet_architectures = [
        "cspresnet50", "cspresnet50d", "cspresnet50w", "cspresnext50",
        "cspdarknet53", "darknet17", "darknet21", "sedarknet21",
        "darknet53", "darknetaa53", "cs3darknet_s", "cs3darknet_m",
        "cs3darknet_l", "cs3darknet_x", "cs3darknet_focus_s",
        "cs3darknet_focus_m", "cs3darknet_focus_l", "cs3darknet_focus_x",
        "cs3sedarknet_l", "cs3sedarknet_x", "cs3sedarknet_xdw",
        "cs3edgenet_x", "cs3se_edgenet_x"
    ]

    if cspnet_type not in cspnet_architectures:
        raise ValueError(f'Unknown CSPNet Architecture: {cspnet_type}')

    try:
        return create_model(cspnet_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{cspnet_type} - Error loading pretrained model: {e}")
        return create_model(cspnet_type, pretrained=False, num_classes=num_classes)
