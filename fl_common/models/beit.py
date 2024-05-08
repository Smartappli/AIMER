from timm import create_model


def get_beit_model(beit_type, num_classes):
    """
    Retrieves a pre-trained BEiT (Bottleneck Transformers) model based on the specified architecture.

    Parameters:
    - beit_type (str): The type of BEiT architecture to be retrieved.
    - num_classes (int): The number of output classes for the classification task.

    Returns:
    - torch.nn.Module: A pre-trained BEiT model with the specified architecture and number of classes.

    Raises:
    - ValueError: If the specified `beit_type` is not one of the supported architectures.
    """
    valid_types = {
        'beit_base_patch16_224', 'beit_base_patch16_384',
        'beit_large_patch16_224', 'beit_large_patch16_384',
        'beit_large_patch16_512', 'beitv2_base_patch16_224',
        'beitv2_large_patch16_224'
    }

    if beit_type not in valid_types:
        raise ValueError(f'Unknown BEiT Architecture: {beit_type}')

    try:
        return create_model(beit_type, pretrained=True, num_classes=num_classes)
    except OSError as e:
        print(f"Error loading pretrained model: {e}")
        return create_model(beit_type, pretrained=False, num_classes=num_classes)
