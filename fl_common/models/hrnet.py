from timm import create_model


def get_hrnet_model(hrnet_type, num_classes):
    """
    Get an HRNet model.

    Parameters:
        hrnet_type (str): Type of HRNet architecture. Options include 'hrnet_w18_small', 'hrnet_w18_small_v2',
                          'hrnet_w18' to 'hrnet_w64', and 'hrnet_w18_ssld' to 'hrnet_w48_ssld'.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: HRNet model.

    Raises:
        ValueError: If an unknown HRNet architecture is specified.
    """
    hrnet_options = [
        "hrnet_w18_small",
        "hrnet_w18_small_v2",
        "hrnet_w18",
        "hrnet_w30",
        "hrnet_w32",
        "hrnet_w40",
        "hrnet_w44",
        "hrnet_w48",
        "hrnet_w64",
        "hrnet_w18_ssld",
        "hrnet_w48_ssld",
    ]

    if hrnet_type not in hrnet_options:
        msg = f"Unknown HRNet Architecture: {hrnet_type}"
        raise ValueError(msg)

    try:
        return create_model(
            hrnet_type, pretrained=True, num_classes=num_classes
        )
    except RuntimeError as e:
        print(f"{hrnet_type} - Error loading pretrained model: {e}")
        return create_model(
            hrnet_type, pretrained=False, num_classes=num_classes
        )
