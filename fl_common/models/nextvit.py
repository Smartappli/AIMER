from timm import create_model


def get_nextvit_model(nextvit_type, num_classes):
    """
    Get a NEXTVIT model based on the specified architecture type.

    Args:
        nextvit_type (str): The type of NEXTVIT architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The NEXTVIT model.

    Raises:
        ValueError: If an unknown NEXTVIT architecture type is specified.
    """
    valid_nextvit_types = ["nextvit_small", "nextvit_base", "nextvit_large"]

    if nextvit_type not in valid_nextvit_types:
        raise ValueError(f"Unknown NEXTVIT Architecture: {nextvit_type}")

    try:
        return create_model(
            nextvit_type, pretrained=True, num_classes=num_classes
        )
    except RuntimeError as e:
        print(f"{nextvit_type} - Error loading pretrained model: {e}")
        return create_model(
            nextvit_type, pretrained=False, num_classes=num_classes
        )
