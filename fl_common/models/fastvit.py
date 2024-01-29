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
        fastvit_model = create_model('fastvit_t8', pretrained=True, num_classes=num_classes)
    elif fastvit_type == 'fastvit_t12':
        fastvit_model = create_model('fastvit_t12', pretrained=True, num_classes=num_classes)
    elif fastvit_type == 'fastvit_s12':
        fastvit_model = create_model('fastvit_s12', pretrained=True, num_classes=num_classes)
    elif fastvit_type == 'fastvit_sa12':
        fastvit_model = create_model('fastvit_sa12', pretrained=True, num_classes=num_classes)
    elif fastvit_type == 'fastvit_sa24':
        fastvit_model = create_model('fastvit_sa24', pretrained=True, num_classes=num_classes)
    elif fastvit_type == 'fastvit_sa36':
        fastvit_model = create_model('fastvit_sa36', pretrained=True, num_classes=num_classes)
    elif fastvit_type == 'fastvit_ma36':
        fastvit_model = create_model('fastvit_ma36', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Fastvit Architecture: {fastvit_type}')

    return fastvit_model
