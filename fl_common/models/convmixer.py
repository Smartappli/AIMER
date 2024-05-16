from timm import create_model


def get_convmixer_model(convmixer_type, num_classes):
    """
    Creates and returns a Convmixer model based on the specified architecture type.

    Parameters:
        convmixer_type (str): The type of Convmixer architecture to use.
        num_classes (int): The number of classes for the final classification layer.

    Returns:
        torch.nn.Module: The Convmixer model with the specified architecture and number of classes.

    Raises:
        ValueError: If the specified Convmixer architecture type is unknown.
    """
    valid_types = [
        "convmixer_1536_20",
        "convmixer_768_32",
        "convmixer_1024_20_ks9_p14"]

    if convmixer_type not in valid_types:
        raise ValueError(f"Unknown Convmixer Architecture: {convmixer_type}")

    try:
        return create_model(
            convmixer_type,
            pretrained=True,
            num_classes=num_classes)
    except RuntimeError as e:
        print(f"{convmixer_type} - Error loading pretrained model: {e}")
        return create_model(
            convmixer_type,
            pretrained=False,
            num_classes=num_classes)
