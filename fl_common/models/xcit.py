from timm import create_model


def get_xcit_model(xcit_type, num_classes):
    """
    Create and return an XCiT model based on the specified architecture type and number of classes.

    Parameters:
    - xcit_type (str): Type of XCiT architecture. Supported types:
        - 'xcit_nano_12_p16_224'
        - 'xcit_nano_12_p16_384'
        - 'xcit_tiny_12_p16_224'
        - 'xcit_tiny_12_p16_384'
        - 'xcit_small_12_p16_224'
        - 'xcit_small_12_p16_384'
        - 'xcit_tiny_24_p16_224'
        - 'xcit_tiny_24_p16_384'
        - 'xcit_small_24_p16_224'
        - 'xcit_small_24_p16_384'
        - 'xcit_medium_24_p16_224'
        - 'xcit_medium_24_p16_384'
        - 'xcit_large_24_p16_224'
        - 'xcit_large_24_p16_384'
        - 'xcit_nano_12_p8_224'
        - 'xcit_nano_12_p8_384'
        - 'xcit_tiny_12_p8_224'
        - 'xcit_tiny_12_p8_384'
        - 'xcit_small_12_p8_224'
        - 'xcit_small_12_p8_384'
        - 'xcit_tiny_24_p8_224'
        - 'xcit_tiny_24_p8_384'
        - 'xcit_small_24_p8_224'
        - 'xcit_small_24_p8_384'
        - 'xcit_medium_24_p8_224'
        - 'xcit_medium_24_p8_384'
        - 'xcit_large_24_p8_224'
        - 'xcit_large_24_p8_384'
    - num_classes (int): Number of output classes for the model.

    Returns:
    - xcit_model: A pre-trained XCiT model with the specified architecture and number of classes.

    Raises:
    - ValueError: If the provided `xcit_type` is not recognized.
    """
    if xcit_type == 'xcit_nano_12_p16_224':
        xcit_model = create_model('xcit_nano_12_p16_224', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_nano_12_p16_384':
        xcit_model = create_model('xcit_nano_12_p16_384', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_tiny_12_p16_224':
        xcit_model = create_model('xcit_tiny_12_p16_224', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_tiny_12_p16_384':
        xcit_model = create_model('xcit_tiny_12_p16_384', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_small_12_p16_224':
        xcit_model = create_model('xcit_small_12_p16_224', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_small_12_p16_384':
        xcit_model = create_model('xcit_small_12_p16_384', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_tiny_24_p16_224':
        xcit_model = create_model('xcit_tiny_24_p16_224', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_tiny_24_p16_384':
        xcit_model = create_model('xcit_tiny_24_p16_384', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_small_24_p16_224':
        xcit_model = create_model('xcit_small_24_p16_224', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_small_24_p16_384':
        xcit_model = create_model('xcit_small_24_p16_384', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_medium_24_p16_224':
        xcit_model = create_model('xcit_medium_24_p16_224', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_medium_24_p16_384':
        xcit_model = create_model('xcit_medium_24_p16_384', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_large_24_p16_224':
        xcit_model = create_model('xcit_large_24_p16_224', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_large_24_p16_384':
        xcit_model = create_model('xcit_large_24_p16_384', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_nano_12_p8_224':
        xcit_model = create_model('xcit_nano_12_p8_224', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_nano_12_p8_384':
        xcit_model = create_model('xcit_nano_12_p8_384', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_tiny_12_p8_224':
        xcit_model = create_model('xcit_tiny_12_p8_224', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_tiny_12_p8_384':
        xcit_model = create_model('xcit_tiny_12_p8_384', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_small_12_p8_224':
        xcit_model = create_model('xcit_small_12_p8_224', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_small_12_p8_384':
        xcit_model = create_model('xcit_small_12_p8_384', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_tiny_24_p8_224':
        xcit_model = create_model('xcit_tiny_24_p8_224', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_tiny_24_p8_384':
        xcit_model = create_model('xcit_tiny_24_p8_384', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_small_24_p8_224':
        xcit_model = create_model('xcit_small_24_p8_224', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_small_24_p8_384':
        xcit_model = create_model('xcit_small_24_p8_384', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_medium_24_p8_224':
        xcit_model = create_model('xcit_medium_24_p8_224', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_medium_24_p8_384':
        xcit_model = create_model('xcit_medium_24_p8_384', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_large_24_p8_224':
        xcit_model = create_model('xcit_large_24_p8_224', pretrained=True, num_classes=num_classes)
    elif xcit_type == 'xcit_large_24_p8_384':
        xcit_model = create_model('xcit_large_24_p8_384', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Xcit Architecture : {xcit_type}')

    return xcit_model
