from timm import create_model


def get_nfnet_model(nfnet_type, num_classes):
    """
    Get an NFNet model based on the specified architecture type.

    Args:
        nfnet_type (str): The type of NFNet architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The NFNet model.

    Raises:
        ValueError: If an unknown NFNet architecture type is specified.
    """
    valid_nfnet_types = [
        'dm_nfnet_f0', 'dm_nfnet_f1', 'dm_nfnet_f2', 'dm_nfnet_f3',
        'dm_nfnet_f4', 'dm_nfnet_f5', 'dm_nfnet_f6', 'nfnet_f0',
        'nfnet_f1', 'nfnet_f2', 'nfnet_f3', 'nfnet_f4', 'nfnet_f5',
        'nfnet_f6', 'nfnet_f7', 'nfnet_l0', 'eca_nfnet_l0', 'eca_nfnet_l1',
        'eca_nfnet_l2', 'eca_nfnet_l3', 'nf_regnet_b0', 'nf_regnet_b1',
        'nf_regnet_b2', 'nf_regnet_b3', 'nf_regnet_b4', 'nf_regnet_b5',
        'nf_resnet26', 'nf_resnet50', 'nf_resnet101', 'nf_seresnet26',
        'nf_seresnet50', 'nf_seresnet101', 'nf_ecaresnet26', 'nf_ecaresnet50',
        'nf_ecaresnet101'
    ]

    if nfnet_type not in valid_nfnet_types:
        raise ValueError(f'Unknown NFNet Architecture: {nfnet_type}')

    try:
        return create_model(
            nfnet_type,
            pretrained=True,
            num_classes=num_classes)
    except RuntimeError as e:
        print(f"{nfnet_type} - Error loading pretrained model: {e}")
        return create_model(
            nfnet_type,
            pretrained=False,
            num_classes=num_classes)
