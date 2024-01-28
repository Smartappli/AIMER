from timm import create_model


def get_davit_model(davit_type, num_classes):
    """
    Create and return a Davit model based on the specified architecture.

    Parameters:
    - davit_type (str): Type of Davit architecture ('davit_tiny', 'davit_small', 'davit_base', 'davit_large', 'davit_hurge', 'davit_giant').
    - num_classes (int): Number of output classes.

    Returns:
    - davit_model: Created Davit model.
    """

    # Check the value of davit_type and create the corresponding Davit model
    if davit_type == "davit_tiny":
        davit_model = create_model('davit_tiny', pretrained=True, num_classes=num_classes)
    elif davit_type == "davit_small":
        davit_model = create_model('davit_small', pretrained=True, num_classes=num_classes)
    elif davit_type == "davit_base":
        davit_model = create_model('davit_base', pretrained=True, num_classes=num_classes)
    elif davit_type == "davit_large":
        try:
            davit_model = create_model('davit_large', pretrained=True, num_classes=num_classes)
        except:
            davit_model = create_model('davit_large', pretrained=False, num_classes=num_classes)
    elif davit_type == "davit_huge":
        try:
            davit_model = create_model('davit_huge', pretrained=True, num_classes=num_classes)
        except:
            davit_model = create_model('davit_huge', pretrained=False, num_classes=num_classes)
    elif davit_type == "davit_giant":
        try:
            davit_model = create_model('davit_giant', pretrained=True, num_classes=num_classes)
        except:
            davit_model = create_model('davit_giant', pretrained=False, num_classes=num_classes)
    else:
        # Raise a ValueError if an unknown Davit architecture is specified
        raise ValueError(f'Unknown Davit Architecture: {davit_type}')

    # Return the created Davit model
    return davit_model
