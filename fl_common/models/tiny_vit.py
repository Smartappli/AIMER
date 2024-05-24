from timm import create_model


def get_tiny_vit_model(tiny_vit_type, num_classes):
    """
    Create and return a Tiny Vision Transformer (TinyViT) model based on the specified architecture.

    Parameters:
    - tiny_vit_type (str): Type of TinyViT architecture ('tiny_vit_5m_224', 'tiny_vit_11m_224', 'tiny_vit_21m_224',
      'tiny_vit_21m_384', 'tiny_vit_21m_512').
    - num_classes (int): Number of output classes.

    Returns:
    - tiny_vit_model: Created TinyViT model.

    Raises:
    - ValueError: If an unknown TinyViT architecture is specified.
    """

    supported_types = {
        "tiny_vit_5m_224",
        "tiny_vit_11m_224",
        "tiny_vit_21m_224",
        "tiny_vit_21m_384",
        "tiny_vit_21m_512",
    }

    if tiny_vit_type not in supported_types:
        raise ValueError(f"Unknown TinyViT Architecture: {tiny_vit_type}")

    try:
        return create_model(
            tiny_vit_type, pretrained=True, num_classes=num_classes
        )
    except RuntimeError as e:
        print(f"{tiny_vit_type} - Error loading pretrained model: {e}")
        return create_model(
            tiny_vit_type, pretrained=False, num_classes=num_classes
        )
