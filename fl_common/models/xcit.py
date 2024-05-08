from timm import create_model


def get_xcit_model(xcit_type, num_classes):
    """
    Create and return an XCiT model based on the specified architecture type and number of classes.
    """
    supported_types = [
        'xcit_nano_12_p16_224', 'xcit_nano_12_p16_384',
        'xcit_tiny_12_p16_224', 'xcit_tiny_12_p16_384',
        'xcit_small_12_p16_224', 'xcit_small_12_p16_384',
        'xcit_tiny_24_p16_224', 'xcit_tiny_24_p16_384',
        'xcit_small_24_p16_224', 'xcit_small_24_p16_384',
        'xcit_medium_24_p16_224', 'xcit_medium_24_p16_384',
        'xcit_large_24_p16_224', 'xcit_large_24_p16_384',
        'xcit_nano_12_p8_224', 'xcit_nano_12_p8_384',
        'xcit_tiny_12_p8_224', 'xcit_tiny_12_p8_384',
        'xcit_small_12_p8_224', 'xcit_small_12_p8_384',
        'xcit_tiny_24_p8_224', 'xcit_tiny_24_p8_384',
        'xcit_small_24_p8_224', 'xcit_small_24_p8_384',
        'xcit_medium_24_p8_224', 'xcit_medium_24_p8_384',
        'xcit_large_24_p8_224', 'xcit_large_24_p8_384'
    ]

    if xcit_type not in supported_types:
        raise ValueError(f"Unsupported XCiT type: {xcit_type}")

    try:
        xcit_model = create_model(xcit_type, pretrained=True, num_classes=num_classes)
    except OSError:
        xcit_model = create_model(xcit_type, pretrained=False, num_classes=num_classes)

    return xcit_model
