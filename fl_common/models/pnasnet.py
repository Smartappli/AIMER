from timm import create_model


def get_pnasnet_model(pnasnet_type, num_classes):
    """
    Get a PNASNet model based on the specified architecture type.
    """
    # Validate the architecture type
    valid_types = ['pnasnet5large']
    if pnasnet_type not in valid_types:
        raise ValueError(f'Unknown Pnasnet Architecture: {pnasnet_type}')

    # Attempt to create the model with pretrained weights
    try:
        return create_model(pnasnet_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{pnasnet_type} - Error loading pretrained model: {e}")
        return create_model(pnasnet_type, pretrained=False, num_classes=num_classes)
