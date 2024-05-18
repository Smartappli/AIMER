from timm import create_model


def get_dpn_model(dpn_type, num_classes):
    """
    Creates and returns a DPN (Dual-Path Network) model based on the specified architecture type.

    Parameters:
        dpn_type (str): Type of DPN architecture to use. Options: "dpn48b", "dpn68", "dpn68b", "dpn92", "dpn98", "dpn131", "dpn107".
        num_classes (int): Number of output classes.

    Returns:
        dpn_model: A DPN model instance based on the specified architecture type.

    Raises:
        ValueError: If an unknown DPN architecture type is specified.
    """
    dpn_options = ["dpn48b", "dpn68", "dpn68b", "dpn92", "dpn98", "dpn131", "dpn107"]
    if dpn_type not in dpn_options:
        raise ValueError(f"Unknown DPN Architecture: {dpn_type}")

    try:
        return create_model(dpn_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{dpn_type} - Error loading pretrained model: {e}")
        return create_model(dpn_type, pretrained=False, num_classes=num_classes)
