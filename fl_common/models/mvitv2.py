from timm import create_model


def get_mvitv2_model(mvitv2_type, num_classes):
    """
    Create and return an instance of the specified MViTv2 architecture.

    Args:
        mvitv2_type (str): The type of MViTv2 architecture to create. Options include:
                           'mvitv2_tiny', 'mvitv2_small', 'mvitv2_base', 'mvitv2_large',
                           'mvitv2_small_cls', 'mvitv2_base_cls', 'mvitv2_large_cls', 'mvitv2_huge_cls'.
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The created instance of the specified MViTv2 architecture.

    Raises:
        ValueError: If an unknown MViTv2 architecture type is specified.
    """
    mvitv2_options = [
        "mvitv2_tiny",
        "mvitv2_small",
        "mvitv2_base",
        "mvitv2_large",
        "mvitv2_small_cls",
        "mvitv2_base_cls",
        "mvitv2_large_cls",
        "mvitv2_huge_cls",
    ]

    if mvitv2_type not in mvitv2_options:
        raise ValueError(f"Unknown MViTv2 Architecture: {mvitv2_type}")

    try:
        return create_model(
            mvitv2_type,
            pretrained=True,
            num_classes=num_classes)
    except RuntimeError as e:
        print(f"{mvitv2_type} - Error loading pretrained model: {e}")
        return create_model(
            mvitv2_type,
            pretrained=False,
            num_classes=num_classes)
