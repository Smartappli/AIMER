from timm import create_model


def get_nfnet_model(nfnet_type, num_classes):
    """
    Get an NFNet model based on the specified architecture type.

    Args:
        nfnet_type (str): The type of NFNet architecture. It can be one of the following:
            - 'dm_nfnet_f0': NFNet-F0 architecture by DeiT.
            - 'dm_nfnet_f1': NFNet-F1 architecture by DeiT.
            - 'dm_nfnet_f2': NFNet-F2 architecture by DeiT.
            - 'dm_nfnet_f3': NFNet-F3 architecture by DeiT.
            - 'dm_nfnet_f4': NFNet-F4 architecture by DeiT.
            - 'dm_nfnet_f5': NFNet-F5 architecture by DeiT.
            - 'dm_nfnet_f6': NFNet-F6 architecture by DeiT.
            - 'nfnet_f0': NFNet-F0 architecture.
            - 'nfnet_f1': NFNet-F1 architecture.
            - 'nfnet_f2': NFNet-F2 architecture.
            - 'nfnet_f3': NFNet-F3 architecture.
            - 'nfnet_f4': NFNet-F4 architecture.
            - 'nfnet_f5': NFNet-F5 architecture.
            - 'nfnet_f6': NFNet-F6 architecture.
            - 'nfnet_f7': NFNet-F7 architecture.
            - 'nfnet_l0': NFNet-L0 architecture.
            - 'eca_nfnet_l0': NFNet-L0 architecture with ECA attention.
            - 'eca_nfnet_l1': NFNet-L1 architecture with ECA attention.
            - 'eca_nfnet_l2': NFNet-L2 architecture with ECA attention.
            - 'eca_nfnet_l3': NFNet-L3 architecture with ECA attention.
            - 'nf_regnet_b0': NF-RegNet-B0 architecture.
            - 'nf_regnet_b1': NF-RegNet-B1 architecture.
            - 'nf_regnet_b2': NF-RegNet-B2 architecture.
            - 'nf_regnet_b3': NF-RegNet-B3 architecture.
            - 'nf_regnet_b4': NF-RegNet-B4 architecture.
            - 'nf_regnet_b5': NF-RegNet-B5 architecture.
            - 'nf_resnet26': NF-ResNet-26 architecture.
            - 'nf_resnet50': NF-ResNet-50 architecture.
            - 'nf_resnet101': NF-ResNet-101 architecture.
            - 'nf_seresnet26': NF-SE-ResNet-26 architecture.
            - 'nf_seresnet50': NF-SE-ResNet-50 architecture.
            - 'nf_seresnet101': NF-SE-ResNet-101 architecture.
            - 'nf_ecaresnet26': NF-ECA-ResNet-26 architecture.
            - 'nf_ecaresnet50': NF-ECA-ResNet-50 architecture.
            - 'nf_ecaresnet101': NF-ECA-ResNet-101 architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The NFNet model.

    Raises:
        ValueError: If an unknown NFNet architecture type is specified.
    """
    if nfnet_type == 'dm_nfnet_f0':
        nfnet_model = create_model('dm_nfnet_f0',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'dm_nfnet_f1':
        nfnet_model = create_model('dm_nfnet_f1',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'dm_nfnet_f2':
        nfnet_model = create_model('dm_nfnet_f2',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'dm_nfnet_f3':
        nfnet_model = create_model('dm_nfnet_f1',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'dm_nfnet_f4':
        nfnet_model = create_model('dm_nfnet_f4',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'dm_nfnet_f5':
        nfnet_model = create_model('dm_nfnet_f5',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'dm_nfnet_f6':
        nfnet_model = create_model('dm_nfnet_f6',pretrained=True,num_classes=num_classes)
    elif nfnet_type == 'nfnet_f0':
        nfnet_model = create_model('nfnet_f0', pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nfnet_f1':
        nfnet_model = create_model('nfnet_f1', pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nfnet_f2':
        nfnet_model = create_model('nfnet_f2', pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nfnet_f3':
        nfnet_model = create_model('nfnet_f3', pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nfnet_f4':
        nfnet_model = create_model('nfnet_f4', pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nfnet_f5':
        nfnet_model = create_model('nfnet_f5', pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nfnet_f6':
        nfnet_model = create_model('nfnet_f6', pretrained=True,num_classes=num_classes)
    elif nfnet_type == 'nfnet_f7':
        nfnet_model = create_model('nfnet_f7', pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nfnet_l0':
        nfnet_model = create_model('nfnet_l0', pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'eca_nfnet_l0':
        nfnet_model = create_model('eca_nfnet_l0', pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'eca_nfnet_l1':
        nfnet_model = create_model('eca_nfnet_l1', pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'eca_nfnet_l2':
        nfnet_model = create_model('eca_nfnet_l2', pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'eca_nfnet_l3':
        nfnet_model = create_model('eca_nfnet_l3',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nf_regnet_b0':
        nfnet_model = create_model('nf_regnet_b0',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nf_regnet_b1':
        nfnet_model = create_model('nf_regnet_b1',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nf_regnet_b2':
        nfnet_model = create_model('nf_regnet_b2', pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nf_regnet_b3':
        nfnet_model = create_model('nf_regnet_b3',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nf_regnet_b4':
        nfnet_model = create_model('nf_regnet_b4',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nf_regnet_b5':
        nfnet_model = create_model('nf_regnet_b5',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nf_resnet26':
        nfnet_model = create_model('nf_resnet26',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nf_resnet50':
        nfnet_model = create_model('nf_resnet50',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nf_resnet101':
        nfnet_model = create_model('nf_resnet101',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nf_seresnet26':
        nfnet_model = create_model('nf_seresnet26',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nf_seresnet50':
        nfnet_model = create_model('nf_seresnet50',pretrained=True,num_classes=num_classes)
    elif nfnet_type == 'nf_seresnet101':
        nfnet_model = create_model('nf_seresnet101', pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nf_ecaresnet26':
        nfnet_model = create_model('nf_ecaresnet26',pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nf_ecaresnet50':
        nfnet_model = create_model('nf_ecaresnet50', pretrained=True, num_classes=num_classes)
    elif nfnet_type == 'nf_ecaresnet101':
        nfnet_model = create_model('nf_ecaresnet101', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Nfnet Architecture: {nfnet_type}')

    return nfnet_model