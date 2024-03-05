from timm import create_model


def get_pvt_v2_model(pvt_v2_type, num_classes):
    """
    Get a PVTv2 model based on the specified architecture type.

    Args:
        pvt_v2_type (str): The type of PVTv2 architecture. It can be one of the following:
            - 'pvt_v2_b0': PVTv2 architecture variant B0.
            - 'pvt_v2_b1': PVTv2 architecture variant B1.
            - 'pvt_v2_b2': PVTv2 architecture variant B2.
            - 'pvt_v2_b3': PVTv2 architecture variant B3.
            - 'pvt_v2_b4': PVTv2 architecture variant B4.
            - 'pvt_v2_b5': PVTv2 architecture variant B5.
            - 'pvt_v2_b2_li': PVTv2 architecture variant B2 with linear initialization.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The PVTv2 model.

    Raises:
        ValueError: If an unknown PVTv2 architecture type is specified.
    """
    if pvt_v2_type == 'pvt_v2_b0':
        try:
            pvt_v2_model = create_model('pvt_v2_b0',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            pvt_v2_model = create_model('pvt_v2_b0',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif pvt_v2_type == 'pvt_v2_b1':
        try:
            pvt_v2_model = create_model('pvt_v2_b1',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            pvt_v2_model = create_model('pvt_v2_b1',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif pvt_v2_type == 'pvt_v2_b2':
        try:
            pvt_v2_model = create_model('pvt_v2_b2',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            pvt_v2_model = create_model('pvt_v2_b2',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif pvt_v2_type == 'pvt_v2_b3':
        try:
            pvt_v2_model = create_model('pvt_v2_b3',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            pvt_v2_model = create_model('pvt_v2_b3',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif pvt_v2_type == 'pvt_v2_b4':
        try:
            pvt_v2_model = create_model('pvt_v2_b4',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            pvt_v2_model = create_model('pvt_v2_b4',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif pvt_v2_type == 'pvt_v2_b5':
        try:
            pvt_v2_model = create_model('pvt_v2_b5',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            pvt_v2_model = create_model('pvt_v2_b5',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif pvt_v2_type == 'pvt_v2_b2_li':
        try:
            pvt_v2_model = create_model('pvt_v2_b2_li',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            pvt_v2_model = create_model('pvt_v2_b2_li',
                                        pretrained=False,
                                        num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Pvt_v2 Architecture: {pvt_v2_type}')

    return pvt_v2_model
