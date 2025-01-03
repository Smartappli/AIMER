from timm import create_model


def get_eva_model(eva_type, num_classes):
    """
    Get an EVA (Efficient Vision Architecture) model.

    Parameters:
        eva_type (str): Type of EVA architecture. Options include:
            - "eva_giant_patch14_224"
            - "eva_giant_patch14_336"
            - "eva_giant_patch14_560"
            - "eva02_tiny_patch14_224"
            - "eva02_small_patch14_224"
            - "eva02_base_patch14_224"
            - "eva02_large_patch14_224"
            - "eva02_tiny_patch14_336"
            - "eva02_small_patch14_336"
            - "eva02_base_patch14_448"
            - "eva02_large_patch14_448"
            - "eva_giant_patch14_clip_224"
            - "eva02_base_patch16_clip_224"
            - "eva02_large_patch14_clip_224"
            - "eva02_large_patch14_clip_336"
            - "eva02_enormous_patch14_clip_224"
            - "vit_medium_patch16_rope_reg1_gap_256"
            - "vit_mediumd_patch16_rope_reg1_gap_256"
            - "vit_betwixt_patch16_rope_reg4_gap_256"
            - "vit_base_patch16_rope_reg1_gap_256"

        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: EVA model.

    Raises:
        ValueError: If an unknown EVA architecture is specified.
    """
    supported_types = {
        "eva_giant_patch14_224",
        "eva_giant_patch14_336",
        "eva_giant_patch14_560",
        "eva02_tiny_patch14_224",
        "eva02_small_patch14_224",
        "eva02_base_patch14_224",
        "eva02_large_patch14_224",
        "eva02_tiny_patch14_336",
        "eva02_small_patch14_336",
        "eva02_base_patch14_448",
        "eva02_large_patch14_448",
        "eva_giant_patch14_clip_224",
        "eva02_base_patch16_clip_224",
        "eva02_large_patch14_clip_224",
        "eva02_large_patch14_clip_336",
        "eva02_enormous_patch14_clip_224",
        "vit_medium_patch16_rope_reg1_gap_256",
        "vit_mediumd_patch16_rope_reg1_gap_256",
        "vit_betwixt_patch16_rope_reg4_gap_256",
        "vit_base_patch16_rope_reg1_gap_256",
    }

    if eva_type not in supported_types:
        msg = f"Unknown EVA Architecture: {eva_type}"
        raise ValueError(msg)

    try:
        return create_model(eva_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{eva_type} - Error loading pretrained model: {e}")
        return create_model(eva_type, pretrained=False, num_classes=num_classes)
