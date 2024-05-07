from timm import create_model


def get_fastvit_model(fastvit_type, num_classes):
    """
    Create and return a FastViT model based on the specified architecture type and number of classes.

    Parameters:
    - fastvit_type (str): Type of FastViT architecture to be used.
    - num_classes (int): Number of output classes for the model.

    Returns:
    - fastvit_model: Created FastViT model.

    Raises:
    - ValueError: If an unknown FastViT architecture type is provided.
    """
    if fastvit_type == 'fastvit_t8':
        try:
            fastvit_model = create_model('fastvit_t8',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            fastvit_model = create_model('fastvit_t8',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif fastvit_type == 'fastvit_t12':
        try:
            fastvit_model = create_model('fastvit_t12',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            fastvit_model = create_model('fastvit_t12',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif fastvit_type == 'fastvit_s12':
        try:
            fastvit_model = create_model('fastvit_s12',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            fastvit_model = create_model('fastvit_s12',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif fastvit_type == 'fastvit_sa12':
        try:
            fastvit_model = create_model('fastvit_sa12',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            fastvit_model = create_model('fastvit_sa12',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif fastvit_type == 'fastvit_sa24':
        try:
            fastvit_model = create_model('fastvit_sa24',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            fastvit_model = create_model('fastvit_sa24',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif fastvit_type == 'fastvit_sa36':
        try:
            fastvit_model = create_model('fastvit_sa36',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            fastvit_model = create_model('fastvit_sa36',
                                         pretrained=False,
                                         num_classes=num_classes)
    elif fastvit_type == 'fastvit_ma36':
        try:
            fastvit_model = create_model('fastvit_ma36',
                                         pretrained=True,
                                         num_classes=num_classes)
        except Exception:
            fastvit_model = create_model('fastvit_ma36',
                                         pretrained=False,
                                         num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Fastvit Architecture: {fastvit_type}')

    return fastvit_model
