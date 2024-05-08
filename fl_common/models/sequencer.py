from timm import create_model


def get_sequencer_model(sequencer_type, num_classes):
    """
    Get a sequencer model based on the specified type.

    Args:
        sequencer_type (str): The type of sequencer architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The sequencer model.

    Raises:
        ValueError: If an unknown sequencer architecture type is specified.
    """
    valid_types = {
        'sequencer2d_s', 'sequencer2d_m', 'sequencer2d_l'
    }

    if sequencer_type not in valid_types:
        raise ValueError(f'Unknown Sequencer Architecture: {sequencer_type}')

    try:
        return create_model(sequencer_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"Error loading pretrained model: {e}")
        return create_model(sequencer_type, pretrained=False, num_classes=num_classes)
