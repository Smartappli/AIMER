from timm import create_model


def get_gcvit_model(gcvit_type, num_classes):
    """
    Create and return a GCVIT model based on the specified architecture.

    Parameters:
    - gcvit_type (str): Type of GCVIT architecture ('gcvit_xxtiny', 'gcvit_xtiny', 'gcvit_tiny',
                       'gcvit_small', 'gcvit_base').
    - num_classes (int): Number of output classes.

    Returns:
    - gcvit_model: Created GCVIT model.

    Raises:
    - ValueError: If an unknown GCVIT architecture is specified.
    """
    if gcvit_type == 'gcvit_xxtiny':
        try:
            gcvit_model = create_model('gcvit_xxtiny',
                                       pretrained=True,
                                       num_classes=num_classes)
        except OSError:
            gcvit_model = create_model('gcvit_xxtiny',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif gcvit_type == 'gcvit_xtiny':
        try:
            gcvit_model = create_model('gcvit_xtiny',
                                       pretrained=True,
                                       num_classes=num_classes)
        except OSError:
            gcvit_model = create_model('gcvit_xtiny',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif gcvit_type == 'gcvit_tiny':
        try:
            gcvit_model = create_model('gcvit_tiny',
                                       pretrained=True,
                                       num_classes=num_classes)
        except OSError:
            gcvit_model = create_model('gcvit_tiny',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif gcvit_type == 'gcvit_small':
        try:
            gcvit_model = create_model('gcvit_small',
                                       pretrained=True,
                                       num_classes=num_classes)
        except OSError:
            gcvit_model = create_model('gcvit_small',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif gcvit_type == 'gcvit_base':
        try:
            gcvit_model = create_model('gcvit_base',
                                       pretrained=True,
                                       num_classes=num_classes)
        except OSError:
            gcvit_model = create_model('gcvit_base',
                                       pretrained=False,
                                       num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Gcvit Architecture: {gcvit_type}')

    return gcvit_model
