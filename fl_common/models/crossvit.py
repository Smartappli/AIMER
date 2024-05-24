from timm import create_model


def get_crossvit_model(crossvit_type, num_classes):
    """
    Creates and returns a Crossvit model based on the specified architecture type.

    Parameters:
        crossvit_type (str): The type of Crossvit architecture to use.
        num_classes (int): The number of classes for the final classification layer.

    Returns:
        torch.nn.Module: The Crossvit model with the specified architecture and number of classes.

    Raises:
        ValueError: If the specified Crossvit architecture type is unknown.
    """
    crossvit_architectures = [
        "crossvit_tiny_240",
        "crossvit_small_240",
        "crossvit_base_240",
        "crossvit_9_240",
        "crossvit_15_240",
        "crossvit_18_240",
        "crossvit_9_dagger_240",
        "crossvit_15_dagger_240",
        "crossvit_15_dagger_408",
        "crossvit_18_dagger_240",
        "crossvit_18_dagger_408",
    ]

    if crossvit_type not in crossvit_architectures:
        msg = f"Unknown Crossvit Architecture: {crossvit_type}"
        raise ValueError(msg)

    try:
        return create_model(
            crossvit_type, pretrained=True, num_classes=num_classes
        )
    except RuntimeError as e:
        print(f"{crossvit_type} - Error loading pretrained model: {e}")
        return create_model(
            crossvit_type, pretrained=False, num_classes=num_classes
        )
