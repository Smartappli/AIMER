from timm import create_model


def get_pit_model(pit_type, num_classes):
    """
    Get a PIT model based on the specified architecture type.

    Args:
        pit_type (str): The type of PIT architecture. It can be one of the following:
            - 'pit_b_224': PIT-B model with input size 224x224.
            - 'pit_s_224': PIT-S model with input size 224x224.
            - 'pit_xs_224': PIT-XS model with input size 224x224.
            - 'pit_ti_224': PIT-TI model with input size 224x224.
            - 'pit_b_distilled_224': Distilled PIT-B model with input size 224x224.
            - 'pit_s_distilled_224': Distilled PIT-S model with input size 224x224.
            - 'pit_xs_distilled_224': Distilled PIT-XS model with input size 224x224.
            - 'pit_ti_distilled_224': Distilled PIT-TI model with input size 224x224.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The PIT model.

    Raises:
        ValueError: If an unknown PIT architecture type is specified.
    """
    if pit_type == 'pit_b_224':
        pit_model = create_model('pit_b_224', pretrained=True, num_classes=num_classes)
    elif pit_type == 'pit_s_224':
        pit_model = create_model('pit_s_224', pretrained=True, num_classes=num_classes)
    elif pit_type == 'pit_xs_224':
        pit_model = create_model('pit_xs_224', pretrained=True, num_classes=num_classes)
    elif pit_type == 'pit_ti_224':
        pit_model = create_model('pit_ti_224', pretrained=True, num_classes=num_classes)
    elif pit_type == 'pit_b_distilled_224':
        pit_model = create_model('pit_b_distilled_224', pretrained=True, num_classes=num_classes)
    elif pit_type == 'pit_s_distilled_224':
        pit_model = create_model('pit_s_distilled_224', pretrained=True, num_classes=num_classes)
    elif pit_type == 'pit_xs_distilled_224':
        pit_model = create_model('pit_xs_distilled_224', pretrained=False, num_classes=num_classes)
    elif pit_type == 'pit_ti_distilled_224':
        pit_model = create_model('pit_ti_distilled_224', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Pit Architecture: {pit_type}')

    return pit_model
