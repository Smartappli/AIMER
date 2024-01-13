from timm import create_model


def get_volo_model(volo_type, num_classes):
    """
    Load a pre-trained Volo model of the specified type.

    Parameters:
    - volo_type (str): Type of Volo architecture. Supported types:
        - 'volo_d1_224'
        - 'volo_d1_384'
        - 'volo_d2_224'
        - 'volo_d2_384'
        - 'volo_d3_224'
        - 'volo_d3_384'
        - 'volo_a4_224'
        - 'volo_a4_448'
        - 'volo_a5_224'
        - 'volo_a5_448'
        - 'volo_a5_512'
    - num_classes (int): Number of output classes for the model.

    Returns:
    - torch.nn.Module: Volo model with the specified architecture and number of classes.

    Raises:
    - ValueError: If an unknown Volo architecture type is provided.
    """
    if volo_type == 'volo_d1_224':
        volo_model = create_model('volo_d1_224', pretrained=True, num_classes=num_classes)
    elif volo_type == 'volo_d1_384':
        volo_model = create_model('volo_d1_384', pretrained=True, num_classes=num_classes)
    elif volo_type == 'volo_d2_224':
        volo_model = create_model('volo_d2_224', pretrained=True, num_classes=num_classes)
    elif volo_type == 'volo_d2_384':
        volo_model = create_model('volo_d2_384', pretrained=True, num_classes=num_classes)
    elif volo_type == 'volo_d3_224':
        volo_model = create_model('volo_d3_224', pretrained=True, num_classes=num_classes)
    elif volo_type == 'volo_d3_448':
        volo_model = create_model('volo_d3_448', pretrained=True, num_classes=num_classes)
    elif volo_type == 'volo_d4_224':
        volo_model = create_model('volo_d4_224', pretrained=True, num_classes=num_classes)
    elif volo_type == 'volo_d4_448':
        volo_model = create_model('volo_d4_448', pretrained=True, num_classes=num_classes)
    elif volo_type == 'volo_d5_224':
        volo_model = create_model('volo_d5_224', pretrained=True, num_classes=num_classes)
    elif volo_type == 'volo_d5_448':
        volo_model = create_model('volo_d5_448', pretrained=True, num_classes=num_classes)
    elif volo_type == 'volo_d5_512':
        volo_model = create_model('volo_d5_512', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Volo Architecture : {volo_type}')

    return volo_model
