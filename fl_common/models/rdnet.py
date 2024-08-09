from timm import create_model


def get_rdnet_model(rdnet_type, num_classes):
    valid_rdnet_types = [
        "dla60_res2net",
        "dla60_res2next",
        "dla34",
    ]

    if rdnet_type not in valid_rdnet_types:
        msg = f"Unknown RDNet Architecture: {rdnet_type}"
        raise ValueError(msg)

    try:
        return create_model(rdnet_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{rdnet_type} - Error loading pretrained model: {e}")
        return create_model(rdnet_type, pretrained=False, num_classes=num_classes)
