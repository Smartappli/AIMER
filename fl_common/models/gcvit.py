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
    supported_types = {
        'gcvit_xxtiny', 'gcvit_xtiny', 'gcvit_tiny',
        'gcvit_small', 'gcvit_base'
    }

    if gcvit_type not in supported_types:
        raise ValueError(f'Unknown GCVIT Architecture: {gcvit_type}')

    try:
        return create_model(
            gcvit_type,
            pretrained=True,
            num_classes=num_classes)
    except RuntimeError as e:
        print(f"{gcvit_type} - Error loading pretrained model: {e}")
        return create_model(
            gcvit_type,
            pretrained=False,
            num_classes=num_classes)
