from timm import create_model


def get_selecsls_model(selecsls_type, num_classes):
    """
    Get a SelecSLS model based on the specified architecture type.

    Args:
        selecsls_type (str): The type of SelecSLS architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The SelecSLS model.

    Raises:
        ValueError: If an unknown SelecSLS architecture type is specified.
    """
    valid_types = {
        "selecsls42",
        "selecsls42b",
        "selecsls60",
        "selecsls60b",
        "selecsls84",
    }

    if selecsls_type not in valid_types:
        raise ValueError(f"Unknown SelecSLS Architecture: {selecsls_type}")

    try:
        return create_model(selecsls_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{selecsls_type} - Error loading pretrained model: {e}")
        return create_model(selecsls_type, pretrained=False, num_classes=num_classes)
