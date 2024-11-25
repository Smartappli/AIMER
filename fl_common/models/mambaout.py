from timm import create_model


def get_mambaout_model(mambaout_type, num_classes):
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

    if beit_type not in valid_types:
        msg = f"Unknown Mambaout Architecture: {mambaout_type}"
        raise ValueError(msg)

    try:
        return create_model(mambaout_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{mambaout_type} - Error loading pretrained model: {e}")
        return create_model(
            mambaoout_type,
            pretrained=False,
            num_classes=num_classes,
        )
