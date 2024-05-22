from timm import create_model


def get_cvt_model(cvt_type, num_classes):
    """
    Creates and returns a cvt model based on the specified architecture type.

    Parameters:
        cvt_type (str): The type of cvt architecture to use.
        num_classes (int): The number of classes for the final classification layer.

    Returns:
        torch.nn.Module: The cvt model with the specified architecture and number of classes.

    Raises:
        ValueError: If the specified cvt architecture type is unknown.
    """
    cvt_architectures = [
        "cvt_13",
        "cvt_21",
        "cvt_w24",
    ]

    if cvt_type not in cvt_architectures:
        raise ValueError(f"Unknown cvt Architecture: {cvt_type}")

    try:
        return create_model(cvt_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{cvt_type} - Error loading pretrained model: {e}")
        return create_model(cvt_type, pretrained=False, num_classes=num_classes)
      
