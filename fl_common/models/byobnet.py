from timm import create_model


def get_byobnet_model(byobnet_type, num_classes):
    """
    Crée un modèle Byobnet selon le type spécifié.

    Args:
        byobnet_type (str): Le type de modèle Byobnet à créer. Doit être l'un des suivants:
            - 'gernet_l'
            - 'gernet_m'
            - 'gernet_s'
            - 'repvgg_a0'
            - 'repvgg_a1'
            - 'repvgg_a2'
            - 'repvgg_b0'
            - 'repvgg_b1'
            - 'repvgg_b1g4'
            - 'repvgg_b2'
            - 'repvgg_b2g4'
            - 'repvgg_b3'
            - 'repvgg_b3g4'
            - 'repvgg_d2se'
            - 'resnet51q'
            - 'resnet61q'
            - 'resnext26ts'
            - 'gcresnext26ts'
            - 'seresnext26ts'
            - 'eca_resnext26ts'
            - 'bat_resnext26ts'
            - 'resnet32ts'
            - 'resnet33ts'
            - 'gcresnet33ts'
            - 'seresnet33ts'
            - 'eca_resnet33ts'
            - 'gcresnet50t'
            - 'gcresnext50ts'
            - 'regnetz_b16'
            - 'regnetz_c16'
            - 'regnetz_d32'
            - 'regnetz_d8'
            - 'regnetz_e8'
            - 'regnetz_b16_evos'
            - 'regnetz_c16_evos'
            - 'regnetz_d8_evos'
            - 'mobileone_s0'
            - 'mobileone_s1'
            - 'mobileone_s2'
            - 'mobileone_s3'
            - 'mobileone_s4'

        num_classes (int): Le nombre de classes pour la tâche de classification.

    Returns:
        torch.nn.Module: Le modèle Byobnet créé avec le nombre de classes spécifié.

    Raises:
        ValueError: Si le type de modèle Byobnet spécifié n'est pas reconnu.
    """
    if byobnet_type == 'gernet_l':
        byobnet_model = create_model('gernet_l', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'gernet_m':
        byobnet_model = create_model('gernet_m', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'gernet_s':
        byobnet_model = create_model('gernet_s', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'repvgg_a0':
        byobnet_model = create_model('repvgg_a0', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'repvgg_a1':
        byobnet_model = create_model('repvgg_a1', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'repvgg_a2':
        byobnet_model = create_model('repvgg_a2', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'repvgg_b0':
        byobnet_model = create_model('repvgg_b0', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'repvgg_b1':
        byobnet_model = create_model('repvgg_b1', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'repvgg_b1g4':
        byobnet_model = create_model('repvgg_b1g4', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'repvgg_b2':
        byobnet_model = create_model('repvgg_b2', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'repvgg_b2g4':
        byobnet_model = create_model('repvgg_b2g4', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'repvgg_b3':
        byobnet_model = create_model('repvgg_b3', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'repvgg_b3g4':
        byobnet_model = create_model('repvgg_b3g4', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'repvgg_d2se':
        byobnet_model = create_model('repvgg_d2se', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'resnet51q':
        byobnet_model = create_model('resnet51q', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'resnet61q':
        byobnet_model = create_model('resnet61q', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'resnext26ts':
        byobnet_model = create_model('resnext26ts', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'gcresnext26ts':
        byobnet_model = create_model('gcresnext26ts', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'seresnext26ts':
        byobnet_model = create_model('seresnext26ts', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'eca_resnext26ts':
        byobnet_model = create_model('eca_resnext26ts', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'bat_resnext26ts':
        byobnet_model = create_model('bat_resnext26ts', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'resnet32ts':
        byobnet_model = create_model('resnet32ts', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'resnet33ts':
        byobnet_model = create_model('resnet33ts', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'gcresnet33ts':
        byobnet_model = create_model('gcresnet33ts', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'seresnet33ts':
        byobnet_model = create_model('seresnet33ts', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'eca_resnet33ts':
        byobnet_model = create_model('eca_resnet33ts', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'gcresnet50t':
        byobnet_model = create_model('gcresnet50t', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'gcresnext50ts':
        byobnet_model = create_model('gcresnext50ts',pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'regnetz_b16':
        byobnet_model = create_model('regnetz_b16', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'regnetz_c16':
        byobnet_model = create_model('regnetz_c16', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'regnetz_d32':
        byobnet_model = create_model('regnetz_d32', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'regnetz_d8':
        byobnet_model = create_model('regnetz_d8', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'regnetz_e8':
        byobnet_model = create_model('regnetz_e8', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'regnetz_b16_evos':
        byobnet_model = create_model('regnetz_b16_evos', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'regnetz_c16_evos':
        byobnet_model = create_model('regnetz_c16_evos', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'regnetz_d8_evos':
        byobnet_model = create_model('regnetz_d8_evos', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'mobileone_s0':
        byobnet_model = create_model('mobileone_s0', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'mobileone_s1':
        byobnet_model = create_model('mobileone_s1', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'mobileone_s2':
        byobnet_model = create_model('mobileone_s2', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'mobileone_s3':
        byobnet_model = create_model('mobileone_s3', pretrained=True, num_classes=num_classes)
    elif byobnet_type == 'mobileone_s4':
        byobnet_model = create_model('mobileone_s4', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Byobnet Architecture: {byobnet_type}')

    return byobnet_model
