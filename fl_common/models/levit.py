from timm import create_model


def get_levit_model(levit_type, num_classes):
    """
    Create and return a Levit model based on the specified architecture.

    Parameters:
        levit_type (str): Type of Levit architecture. Options include various 'levit' and 'levit_conv' models.
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Created Levit model.

    Raises:
        ValueError: If an unknown Levit architecture is specified.
    """
    levit_options = [
        'levit_128s', 'levit_128', 'levit_192', 'levit_256', 'levit_384',
        'levit_384_s8', 'levit_512_s8', 'levit_512', 'levit_256d', 'levit_512d',
        'levit_conv_128s', 'levit_conv_128', 'levit_conv_192', 'levit_conv_256',
        'levit_conv_384', 'levit_conv_384_s8', 'levit_conv_512_s8', 'levit_conv_512',
        'levit_conv_256d', 'levit_conv_512d'
    ]

    if levit_type not in levit_options:
        raise ValueError(f'Unknown Levit Architecture: {levit_type}')

    try:
        return create_model(levit_type, pretrained=True, num_classes=num_classes)
    except OSError as e:
        print(f"Error loading pretrained model: {e}")
        return create_model(levit_type, pretrained=False, num_classes=num_classes)
