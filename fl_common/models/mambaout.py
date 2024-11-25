from timm import create_model

def get_mambaout_model(mambaout_type, num_classes):
    """
    Creates a Mambaout model using the specified architecture and number of classes.

    Args:
        mambaout_type (str): The type of Mambaout architecture to create. 
            Must be one of the following:
            - "mambaout_femto"
            - "mambaout_kobe"
            - "mambaout_tiny"
            - "mambaout_small"
            - "mambaout_base"
            - "mambaout_small_rw"
            - "mambaout_base_short_rw"
            - "mambaout_base_tall_rw"
            - "mambaout_base_wide_rw"
            - "mambaout_base_plus_rw"
            - "test_mambaout"
        num_classes (int): The number of output classes for the model.

    Returns:
        nn.Module: The Mambaout model instance with the specified configuration.

    Raises:
        ValueError: If the `mambaout_type` is not one of the valid architecture types.
        RuntimeError: If the pretrained model cannot be loaded. Falls back to 
                      creating the model without pretraining.

    Notes:
        - The function attempts to load a pretrained model if available. If loading the pretrained
          weights fails due to a `RuntimeError`, it creates the model without pretrained weights.
    """
    valid_types = {
        "mambaout_femto",
        "mambaout_kobe",
        "mambaout_tiny",
        "mambaout_small",
        "mambaout_base",
        "mambaout_small_rw",
        "mambaout_base_short_rw",
        "mambaout_base_tall_rw",
        "mambaout_base_wide_rw",
        "mambaout_base_plus_rw",
        "test_mambaout",
    }

    if mambaout_type not in valid_types:
        msg = f"Unknown Mambaout Architecture: {mambaout_type}"
        raise ValueError(msg)

    try:
        return create_model(mambaout_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{mambaout_type} - Error loading pretrained model: {e}")
        return create_model(
            mambaout_type,
            pretrained=False,
            num_classes=num_classes,
        )
