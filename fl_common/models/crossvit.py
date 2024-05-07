from timm import create_model


def get_crossvit_model(crossvit_type, num_classes):
    """
    Creates and returns a Crossvit model based on the specified architecture type.

    Parameters:
        crossvit_type (str): The type of Crossvit architecture to use. It can be one of the following:
                             - "crossvit_tiny_240"
                             - "crossvit_small_240"
                             - "crossvit_base_240"
                             - "crossvit_9_240"
                             - "crossvit_15_240"
                             - "crossvit_18_240"
                             - "crossvit_9_dagger_240"
                             - "crossvit_15_dagger_240"
                             - "crossvit_15_dagger_408"
                             - "crossvit_18_dagger_240"
                             - "crossvit_18_dagger_408"
        num_classes (int): The number of classes for the final classification layer.

    Returns:
        torch.nn.Module: The Crossvit model with the specified architecture and number of classes.

    Raises:
        ValueError: If the specified Crossvit architecture type is unknown.
    """
    if crossvit_type == "crossvit_tiny_240":
        try:
            crossvit_model = create_model('crossvit_tiny_240',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            crossvit_model = create_model('crossvit_tiny_240',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif crossvit_type == "crossvit_small_240":
        try:
            crossvit_model = create_model('crossvit_small_240',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            crossvit_model = create_model('crossvit_small_240',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif crossvit_type == "crossvit_base_240":
        try:
            crossvit_model = create_model('crossvit_base_240',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            crossvit_model = create_model('crossvit_base_240',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif crossvit_type == "crossvit_9_240":
        try:
            crossvit_model = create_model('crossvit_9_240',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            crossvit_model = create_model('crossvit_9_240',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif crossvit_type == "crossvit_15_240":
        try:
            crossvit_model = create_model('crossvit_15_240',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            crossvit_model = create_model('crossvit_15_240',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif crossvit_type == "crossvit_18_240":
        try:
            crossvit_model = create_model('crossvit_18_240',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            crossvit_model = create_model('crossvit_18_240',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif crossvit_type == "crossvit_9_dagger_240":
        try:
            crossvit_model = create_model('crossvit_9_dagger_240',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            crossvit_model = create_model('crossvit_9_dagger_240',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif crossvit_type == "crossvit_15_dagger_240":
        try:
            crossvit_model = create_model('crossvit_15_dagger_240',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            crossvit_model = create_model('crossvit_15_dagger_240',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif crossvit_type == "crossvit_15_dagger_408":
        try:
            crossvit_model = create_model('crossvit_15_dagger_408',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            crossvit_model = create_model('crossvit_15_dagger_408',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif crossvit_type == "crossvit_18_dagger_240":
        try:
            crossvit_model = create_model('crossvit_18_dagger_240',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            crossvit_model = create_model('crossvit_18_dagger_240',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif crossvit_type == "crossvit_18_dagger_408":
        try:
            crossvit_model = create_model('crossvit_18_dagger_408',
                                          pretrained=True,
                                          num_classes=num_classes)
        except ValueError:
            crossvit_model = create_model('crossvit_18_dagger_408',
                                          pretrained=False,
                                          num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Crossvit Architecture: {crossvit_type}')

    return crossvit_model
