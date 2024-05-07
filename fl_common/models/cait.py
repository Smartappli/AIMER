from timm import create_model


def get_cait_model(cait_type, num_classes):
    """
    Create and return a CAIT model based on the specified architecture.

    Parameters:
    - cait_type (str): Type of CAIT architecture ('cait_xxs24_224', 'cait_xxs24_384', 'cait_xxs36_224',
                      'cait_xxs36_384', 'cait_xs24_384', 'cait_s24_224', 'cait_s24_384', 'cait_s36_224',
                      'cait_m36_224', 'cait_m48_448').
    - num_classes (int): Number of output classes.

    Returns:
    - cait_model: Created CAIT model.

    Raises:
    - ValueError: If an unknown CAIT architecture is specified.
    """
    if cait_type == "cait_xxs24_224":
        try:
            cait_model = create_model('cait_xxs24_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except ValueError:
            cait_model = create_model('cait_xxs24_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif cait_type == "cait_xxs24_384":
        try:
            cait_model = create_model('cait_xxs24_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except ValueError:
            cait_model = create_model('cait_xxs24_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif cait_type == "cait_xxs36_224":
        try:
            cait_model = create_model('cait_xxs36_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except ValueError:
            cait_model = create_model('cait_xxs36_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif cait_type == "cait_xxs36_384":
        try:
            cait_model = create_model('cait_xxs36_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except ValueError:
            cait_model = create_model('cait_xxs36_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif cait_type == "cait_xs24_384":
        try:
            cait_model = create_model('cait_xs24_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except ValueError:
            cait_model = create_model('cait_xs24_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif cait_type == "cait_s24_224":
        try:
            cait_model = create_model('cait_s24_224',
                                      pretrained=True,
                                      num_classes=num_classes)
        except ValueError:
            cait_model = create_model('cait_s24_224',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif cait_type == "cait_s24_384":
        try:
            cait_model = create_model('cait_s24_384',
                                      pretrained=True,
                                      num_classes=num_classes)
        except ValueError:
            cait_model = create_model('cait_s24_384',
                                      pretrained=False,
                                      num_classes=num_classes)
    elif cait_type == "cait_s36_384":
        try:
            cait_model = create_model("cait_s36_384",
                                      pretrained=True,
                                      num_classes=num_classes)
        except ValueError:
            cait_model = create_model("cait_s36_384",
                                      pretrained=False,
                                      num_classes=num_classes)
    elif cait_type == "cait_m36_384":
        try:
            cait_model = create_model("cait_m36_384",
                                      pretrained=True,
                                      num_classes=num_classes)
        except ValueError:
            cait_model = create_model("cait_m36_384",
                                      pretrained=False,
                                      num_classes=num_classes)
    elif cait_type == "cait_m48_448":
        try:
            cait_model = create_model("cait_m48_448",
                                      pretrained=True,
                                      num_classes=num_classes)
        except ValueError:
            cait_model = create_model("cait_m48_448",
                                      pretrained=False,
                                      num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Cait Architecture: {cait_type}')

    return cait_model
