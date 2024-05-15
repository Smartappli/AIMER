from timm import create_model


def get_pit_model(pit_type, num_classes):
    """
    Get a PIT model based on the specified architecture type.

    Args:
        pit_type (str): The type of PIT architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The PIT model.

    Raises:
        ValueError: If an unknown PIT architecture type is specified.
    """
    pit_types = [
        'pit_b_224', 'pit_s_224', 'pit_xs_224', 'pit_ti_224',
        'pit_b_distilled_224', 'pit_s_distilled_224',
        'pit_xs_distilled_224', 'pit_ti_distilled_224'
    ]

    if pit_type not in pit_types:
        raise ValueError(f'Unknown PIT Architecture: {pit_type}')

    try:
        return create_model(pit_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{pit_type} - Error loading pretrained model: {e}")
        return create_model(
            pit_type,
            pretrained=False,
            num_classes=num_classes)
