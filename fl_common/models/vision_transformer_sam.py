from timm import create_model


def get_vision_transformer_sam_model(vision_transformer_sam_type, num_classes):
    """
    Function to get a Vision Transformer SAM (Spatial Attention Module) model of a specified type.

    Parameters:
        vision_transformer_sam_type (str): Type of Vision Transformer SAM model to be used.
        num_classes (int): Number of classes for the classification task.

    Returns:
        vision_transformer_sam_model: Vision Transformer SAM model instance with specified architecture and number of classes.

    Raises:
        ValueError: If the specified vision_transformer_sam_type is not one of the supported architectures.
    """
    valid_types = {
        "samvit_base_patch16",
        "samvit_large_patch16",
        "samvit_huge_patch16",
        "samvit_base_patch16_224",
    }

    if vision_transformer_sam_type not in valid_types:
        msg = f"Unknown Vision Transformer SAM Architecture: {vision_transformer_sam_type}"
        raise ValueError(msg)

    try:
        return create_model(
            vision_transformer_sam_type,
            pretrained=True,
            num_classes=num_classes,
        )
    except RuntimeError as e:
        print(
            f"{vision_transformer_sam_type} - Error loading pretrained model: {e}"
        )
        return create_model(
            vision_transformer_sam_type,
            pretrained=False,
            num_classes=num_classes,
        )
