from timm import create_model


def get_levit_model(levit_type, num_classes):
    """
    Create and return a Levit model based on the specified architecture.

    Parameters:
    - levit_type (str): Type of Levit architecture ('levit_128s', 'levit_128', 'levit_192', 'levit_256', 'levit_384',
                       'levit_384_s8', 'levit_512_s8', 'levit_512', 'levit_256d', 'levit_512d', 'levit_conv_128s',
                       'levit_conv_128', 'levit_conv_192', 'levit_conv_256', 'levit_conv_384', 'levit_conv_384_s8',
                       'levit_conv_512_s8', 'levit_conv_512', 'levit_conv_256d', 'levit_conv_512d').
    - num_classes (int): Number of output classes.

    Returns:
    - levit_model: Created Levit model.

    Raises:
    - ValueError: If an unknown Levit architecture is specified.
    """
    if levit_type == "levit_128s":
        levit_model = create_model('levit_128s', pretrained=True, num_classes=num_classes)
    elif levit_type == "levit_128":
        levit_model = create_model('levit_128', pretrained=True, num_classes=num_classes)
    elif levit_type == "levit_192":
        levit_model = create_model('levit_192', pretrained=True, num_classes=num_classes)
    elif levit_type == "levit_256":
        levit_model = create_model('levit_256', pretrained=True, num_classes=num_classes)
    elif levit_type == "levit_384":
        levit_model = create_model('levit_384',pretrained=True, num_classes=num_classes)
    elif levit_type == "levit_384_s8":
        try:
            levit_model = create_model('levit_384_s8', pretrained=True, num_classes=num_classes)
        except:
            levit_model = create_model('levit_384_s8', pretrained=False, num_classes=num_classes)
    elif levit_type == "levit_512_s8":
        try:
            levit_model = create_model('levit_512_s8',pretrained=True, num_classes=num_classes)
        except:
            levit_model = create_model('levit_512_s8',pretrained=False, num_classes=num_classes)
    elif levit_type == "levit_512":
        try:
            levit_model = create_model('levit_512', pretrained=True, num_classes=num_classes)
        except:
            levit_model = create_model('levit_512', pretrained=False, num_classes=num_classes)
    elif levit_type == "levit_256d":
        try:
            levit_model = create_model('levit_256d', pretrained=True, num_classes=num_classes)
        except:
            levit_model = create_model('levit_256d', pretrained=False, num_classes=num_classes)
    elif levit_type == "levit_512d":
        try:
            levit_model = create_model('levit_512d',pretrained=True, num_classes=num_classes)
        except:
            levit_model = create_model('levit_512d',pretrained=False, num_classes=num_classes)
    elif levit_type == "levit_conv_128s":
        levit_model = create_model('levit_conv_128s', pretrained=True, num_classes=num_classes)
    elif levit_type == "levit_conv_128":
        levit_model = create_model('levit_conv_128', pretrained=True, num_classes=num_classes)
    elif levit_type == "levit_conv_192":
        levit_model = create_model('levit_conv_192', pretrained=True, num_classes=num_classes)
    elif levit_type == "levit_conv_256":
        levit_model = create_model('levit_conv_256', pretrained=True, num_classes=num_classes)
    elif levit_type == "levit_conv_384":
        levit_model = create_model('levit_conv_384',pretrained=True, num_classes=num_classes)
    elif levit_type == "levit_conv_384_s8":
        try:
            levit_model = create_model('levit_conv_384_s8', pretrained=True, num_classes=num_classes)
        except:
            levit_model = create_model('levit_conv_384_s8', pretrained=False, num_classes=num_classes)
    elif levit_type == "levit_conv_512_s8":
        try:
            levit_model = create_model('levit_conv_512_s8', pretrained=True, num_classes=num_classes)
        except:
            levit_model = create_model('levit_conv_512_s8', pretrained=False, num_classes=num_classes)
    elif levit_type == "levit_conv_512":
        try:
            levit_model = create_model('levit_conv_512', pretrained=True, num_classes=num_classes)
        except:
            levit_model = create_model('levit_conv_512', pretrained=False, num_classes=num_classes)
    elif levit_type == "levit_conv_256d":
        try:
            levit_model = create_model('levit_conv_256d', pretrained=True, num_classes=num_classes)
        except:
            levit_model = create_model('levit_conv_256d', pretrained=False, num_classes=num_classes)
    elif levit_type == "levit_conv_512d":
        try:
            levit_model = create_model('levit_conv_512d', pretrained=True, num_classes=num_classes)
        except:
            levit_model = create_model('levit_conv_512d', pretrained=False, num_classes=num_classes)
    else:
        # Raise a ValueError if an unknown Levit architecture is specified
        raise ValueError(f'Unknown Levit Architecture: {levit_type}')

    # Return the created Levit model
    return levit_model
