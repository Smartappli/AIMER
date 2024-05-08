from timm import create_model


def get_byoanet_model(byoanet_type, num_classes):
    """
    Creates a Byoanet model according to the specified type.

    Args:
        byoanet_type (str): The type of Byoanet model to create.
        num_classes (int): The number of classes for the classification task.

    Returns:
        torch.nn.Module: The created Byoanet model with the specified number of classes.

    Raises:
        ValueError: If the specified Byoanet model type is not recognized.
    """
    byoanet_types = [
        'botnet26t_256', 'sebotnet33ts_256', 'botnet50ts_256',
        'eca_botnext26ts_256', 'halonet_h1', 'halonet26t',
        'sehalonet33ts', 'halonet50ts', 'eca_halonext26ts',
        'lambda_resnet26t', 'lambda_resnet50ts', 'lambda_resnet26rpt_256',
        'haloregnetz_b', 'lamhalobotnet50ts_256', 'halo2botnet50ts_256'
    ]

    if byoanet_type not in byoanet_types:
        raise ValueError(f'Unknown Byoanet Architecture: {byoanet_type}')

    try:
        return create_model(byoanet_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"Error loading pretrained model: {e}")
        return create_model(byoanet_type, pretrained=False, num_classes=num_classes)
