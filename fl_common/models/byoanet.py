from timm import create_model


def get_byoanet_model(byoanet_type, num_classes):
    """
    Crée un modèle Byoanet selon le type spécifié.

    Args:
        byoanet_type (str): Le type de modèle Byoanet à créer. Doit être l'un des suivants:
            - 'botnet26t_256'
            - 'sebotnet33ts_256'
            - 'botnet50ts_256'
            - 'eca_botnext26ts_256'
            - 'halonet_h1'
            - 'halonet26t'
            - 'sehalonet33ts'
            - 'halonet50ts'
            - 'eca_halonext26ts'
            - 'lambda_resnet26t'
            - 'lambda_resnet50ts'
            - 'lambda_resnet26rpt_256'
            - 'haloregnetz_b'
            - 'lamhalobotnet50ts_256'
            - 'halo2botnet50ts_256'

        num_classes (int): Le nombre de classes pour la tâche de classification.

    Returns:
        torch.nn.Module: Le modèle Byoanet créé avec le nombre de classes spécifié.

    Raises:
        ValueError: Si le type de modèle Byoanet spécifié n'est pas reconnu.
    """
    if byoanet_type == 'botnet26t_256':
        try:
            byoanet_model = create_model('botnet26t_256',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            byoanet_model = create_model('botnet26t_256',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif byoanet_type == 'sebotnet33ts_256':
        try:
            byoanet_model = create_model('sebotnet33ts_256',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            byoanet_model = create_model('sebotnet33ts_256',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif byoanet_type == 'botnet50ts_256':
        try:
            byoanet_model = create_model('botnet50ts_256',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            byoanet_model = create_model('botnet50ts_256',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif byoanet_type == 'eca_botnext26ts_256':
        try:
            byoanet_model = create_model('eca_botnext26ts_256',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            byoanet_model = create_model('eca_botnext26ts_256',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif byoanet_type == 'halonet_h1':
        try:
            byoanet_model = create_model('halonet_h1',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            byoanet_model = create_model('halonet_h1',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif byoanet_type == 'halonet26t':
        try:
            byoanet_model = create_model('halonet26t',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            byoanet_model = create_model('halonet26t',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif byoanet_type == 'sehalonet33ts':
        try:
            byoanet_model = create_model('sehalonet33ts',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            byoanet_model = create_model('sehalonet33ts',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif byoanet_type == 'halonet50ts':
        try:
            byoanet_model = create_model('halonet50ts',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            byoanet_model = create_model('halonet50ts',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif byoanet_type == 'eca_halonext26ts':
        try:
            byoanet_model = create_model('eca_halonext26ts',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            byoanet_model = create_model('eca_halonext26ts',
                                         pretrained=True,
                                         num_classes=num_classes)
    elif byoanet_type == 'lambda_resnet26t':
        try:
            byoanet_model = create_model('lambda_resnet26t',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            byoanet_model = create_model('lambda_resnet26t',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif byoanet_type == 'lambda_resnet50ts':
        try:
            byoanet_model = create_model('lambda_resnet50ts',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            byoanet_model = create_model('lambda_resnet50ts',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif byoanet_type == 'lambda_resnet26rpt_256':
        try:
            byoanet_model = create_model('lambda_resnet26rpt_256',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            byoanet_model = create_model('lambda_resnet26rpt_256',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif byoanet_type == 'haloregnetz_b':
        try:
            byoanet_model = create_model('haloregnetz_b',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            byoanet_model = create_model('haloregnetz_b',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif byoanet_type == 'lamhalobotnet50ts_256':
        try:
            byoanet_model = create_model('lamhalobotnet50ts_256',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            byoanet_model = create_model('lamhalobotnet50ts_256',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif byoanet_type == 'halo2botnet50ts_256':
        try:
            byoanet_model = create_model('halo2botnet50ts_256',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            byoanet_model = create_model('halo2botnet50ts_256',
                                         pretrained=True,
                                         num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Byoanet Architecture: {byoanet_type}')

    return byoanet_model
