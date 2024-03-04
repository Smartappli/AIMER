from timm import create_model


def get_mvitv2_model(mvitv2_type, num_classes):
    """
    Create and return an instance of the specified MViTv2 architecture.

    Args:
    - mvitv2_type (str): The type of MViTv2 architecture to create. It should be one of the following:
                         'mvitv2_tiny', 'mvitv2_small', 'mvitv2_base', 'mvitv2_large',
                         'mvitv2_small_cls', 'mvitv2_base_cls', 'mvitv2_large_cls', 'mvitv2_huge_cls'.
    - num_classes (int): The number of output classes for the model.

    Returns:
    - torch.nn.Module: The created instance of the specified MViTv2 architecture.

    Raises:
    - ValueError: If an unknown MViTv2 architecture type is specified.
    """
    if mvitv2_type == "mvitv2_tiny":
        mvitv2_model = create_model('mvitv2_tiny', pretrained=True, num_classes=num_classes)
    elif mvitv2_type == "mvitv2_small":
        mvitv2_model = create_model('mvitv2_small', pretrained=True, num_classes=num_classes)
    elif mvitv2_type == "mvitv2_base":
        mvitv2_model = create_model('mvitv2_base', pretrained=True, num_classes=num_classes)
    elif mvitv2_type == "mvitv2_large":
        mvitv2_model = create_model('mvitv2_large', pretrained=True, num_classes=num_classes)
    elif mvitv2_type == "mvitv2_small_cls":
        mvitv2_model = create_model('mvitv2_small_cls', pretrained=False, num_classes=num_classes)
    elif mvitv2_type == "mvitv2_base_cls":
        mvitv2_model = create_model('mvitv2_base_cls', pretrained=True, num_classes=num_classes)
    elif mvitv2_type == "mvitv2_large_cls":
        mvitv2_model = create_model('mvitv2_large_cls', pretrained=True, num_classes=num_classes)
    elif mvitv2_type == 'mvitv2_huge_cls':
        mvitv2_model = create_model('mvitv2_huge_cls', pretrained=True, num_classes=num_classes)
    else:
        # Raise a ValueError if an unknown Mvitv2 architecture is specified
        raise ValueError(f'Unknown Mvitv2 Architecture: {mvitv2_type}')

    # Return the created Mvitv2 model
    return mvitv2_model
