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

    # Check the value of tiny_vit_type and create the corresponding TinyViT model
    if tiny_vit_type == 'tiny_vit_5m_224':
        try:
            tiny_vit_model = create_model('tiny_vit_5m_224',
                                          pretrained=True,
                                          num_classes=num_classes)
        except Exception:
            tiny_vit_model = create_model('tiny_vit_5m_224',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif tiny_vit_type == 'tiny_vit_11m_224':
        try:
            tiny_vit_model = create_model('tiny_vit_11m_224',
                                          pretrained=True,
                                          num_classes=num_classes)
        except Exception:
            tiny_vit_model = create_model('tiny_vit_11m_224',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif tiny_vit_type == 'tiny_vit_21m_224':
        try:
            tiny_vit_model = create_model("tiny_vit_21m_224",
                                          pretrained=True,
                                          num_classes=num_classes)
        except Exception:
            tiny_vit_model = create_model("tiny_vit_21m_224",
                                          pretrained=False,
                                          num_classes=num_classes)
    elif tiny_vit_type == 'tiny_vit_21m_384':
        try:
            tiny_vit_model = create_model("tiny_vit_21m_384",
                                          pretrained=True,
                                          num_classes=num_classes)
        except Exception:
            tiny_vit_model = create_model("tiny_vit_21m_384",
                                          pretrained=False,
                                          num_classes=num_classes)
    elif tiny_vit_type == 'tiny_vit_21m_512':
        try:
            tiny_vit_model = create_model("tiny_vit_21m_512",
                                          pretrained=True,
                                          num_classes=num_classes)
        except Exception:
            tiny_vit_model = create_model("tiny_vit_21m_512",
                                          pretrained=False,
                                          num_classes=num_classes)
    else:
        # Raise a ValueError if an unknown TinyViT architecture is specified
        raise ValueError(f'Unknown TinyViT Architecture: {tiny_vit_type}')

    # Return the created TinyViT model
    return tiny_vit_model
