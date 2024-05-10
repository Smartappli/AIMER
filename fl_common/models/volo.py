from timm import create_model


def get_volo_model(volo_type, num_classes):
    """
    Load a pre-trained Volo model of the specified type.

    Parameters:
    - volo_type (str): Type of Volo architecture.
    - num_classes (int): Number of output classes for the model.

    Returns:
    - torch.nn.Module: Volo model with the specified architecture and number of classes.

    Raises:
    - ValueError: If an unknown Volo architecture type is provided.
    """
    valid_types = {
        'volo_d1_224', 'volo_d1_384', 'volo_d2_224', 'volo_d2_384',
        'volo_d3_224', 'volo_d3_448', 'volo_d4_224', 'volo_d4_448',
        'volo_d5_224', 'volo_d5_448', 'volo_d5_512'
    }

    if volo_type not in valid_types:
        raise ValueError(f'Unknown Volo Architecture: {volo_type}')

    try:
        return create_model(volo_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{volo_type} - Error loading pretrained model: {e}")
        return create_model(volo_type, pretrained=False, num_classes=num_classes)
