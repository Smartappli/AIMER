from timm import create_model


def get_cspnet_model(cspnet_type, num_classes):
    """
    Creates and returns a CSPNet model based on the specified architecture type.

    Parameters:
        cspnet_type (str): The type of CSPNet architecture to use. It can be one of the following:
                           - "cspresnet50"
                           - "cspresnet50d"
                           - "cspresnet50w"
                           - "cspresnext50"
                           - "cspdarknet53"
                           - "darknet17"
                           - "darknet21"
                           - "sedarknet21"
                           - "darknet53"
                           - "darknetaa53"
                           - "cs3darknet_s"
                           - "cs3darknet_m"
                           - "cs3darknet_l"
                           - "cs3darknet_x"
                           - "cs3darknet_focus_s"
                           - "cs3darknet_focus_m"
                           - "cs3darknet_focus_l"
                           - "cs3darknet_focus_x"
                           - "cs3sedarknet_l"
                           - "cs3sedarknet_x"
                           - "cs3sedarknet_xdw"
                           - "cs3edgenet_x"
                           - "cs3se_edgenet_x"
        num_classes (int): The number of classes for the final classification layer.

    Returns:
        torch.nn.Module: The CSPNet model with the specified architecture and number of classes.

    Raises:
        ValueError: If the specified CSPNet architecture type is unknown.
    """
    if cspnet_type == "cspresnet50":
        try:
            cspnet_model = create_model('cspresnet50',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cspresnet50',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cspresnet50d":
        try:
            cspnet_model = create_model('cspresnet50d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cspresnet50d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cspresnet50w":
        try:
            cspnet_model = create_model('cspresnet50w',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cspresnet50w',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cspresnext50":
        try:
            cspnet_model = create_model('cspresnext50',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cspresnext50',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cspdarknet53":
        try:
            cspnet_model = create_model('cspdarknet53',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cspdarknet53',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "darknet17":
        try:
            cspnet_model = create_model('darknet17',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('darknet17',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "darknet21":
        try:
            cspnet_model = create_model('darknet21',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('darknet21',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "sedarknet21":
        try:
            cspnet_model = create_model('sedarknet21',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('sedarknet21',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "darknet53":
        try:
            cspnet_model = create_model('darknet53',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('darknet53',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "darknetaa53":
        try:
            cspnet_model = create_model('darknetaa53',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('darknetaa53',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cs3darknet_s":
        try:
            cspnet_model = create_model('cs3darknet_s',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cs3darknet_s',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cs3darknet_m":
        try:
            cspnet_model = create_model('cs3darknet_m',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cs3darknet_m',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cs3darknet_l":
        try:
            cspnet_model = create_model('cs3darknet_l',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cs3darknet_l',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cs3darknet_x":
        try:
            cspnet_model = create_model('cs3darknet_x',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cs3darknet_x',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cs3darknet_focus_s":
        try:
            cspnet_model = create_model('cs3darknet_focus_s',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cs3darknet_focus_s',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cs3darknet_focus_m":
        try:
            cspnet_model = create_model('cs3darknet_focus_m',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cs3darknet_focus_m',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cs3darknet_focus_l":
        try:
            cspnet_model = create_model('cs3darknet_focus_l',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cs3darknet_focus_l',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cs3darknet_focus_x":
        try:
            cspnet_model = create_model('cs3darknet_focus_x',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cs3darknet_focus_x',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cs3sedarknet_l":
        try:
            cspnet_model = create_model('cs3sedarknet_l',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cs3sedarknet_l',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cs3sedarknet_x":
        try:
            cspnet_model = create_model('cs3sedarknet_x',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cs3sedarknet_x',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cs3sedarknet_xdw":
        try:
            cspnet_model = create_model('cs3sedarknet_xdw',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cs3sedarknet_xdw',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cs3edgenet_x":
        try:
            cspnet_model = create_model('cs3edgenet_x',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cs3edgenet_x',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif cspnet_type == "cs3se_edgenet_x":
        try:
            cspnet_model = create_model('cs3se_edgenet_x',
                                        pretrained=True,
                                        num_classes=num_classes)
        except ValueError:
            cspnet_model = create_model('cs3se_edgenet_x',
                                        pretrained=False,
                                        num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Cspnet Architecture: {cspnet_type}')

    return cspnet_model
