from timm import create_model


def get_sequencer_model(sequencer_type, num_classes):
    """
    Get a sequencer model based on the specified type.

    Args:
        sequencer_type (str): The type of sequencer architecture. It can be one of the following:
            - 'sequencer2d_s': Small sequencer architecture.
            - 'sequencer2d_m': Medium sequencer architecture.
            - 'sequencer2d_l': Large sequencer architecture.
        num_classes (int): The number of output classes.

    Returns:
        torch.nn.Module: The sequencer model.

    Raises:
        ValueError: If an unknown sequencer architecture type is specified.
    """
    if sequencer_type == 'sequencer2d_s':
        sequencer_model = create_model('sequencer2d_s', pretrained=True, num_classes=num_classes)
    elif sequencer_type == 'sequencer2d_m':
        sequencer_model = create_model('sequencer2d_m', pretrained=True, num_classes=num_classes)
    elif sequencer_type == 'sequencer2d_l':
        sequencer_model = create_model('sequencer2d_l', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Sequencer Architecture: {sequencer_type}')

    return sequencer_model