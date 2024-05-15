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
    supported_types = {
        'fastvit_t8', 'fastvit_t12', 'fastvit_s12', 'fastvit_sa12',
        'fastvit_sa24', 'fastvit_sa36', 'fastvit_ma36'
    }

    if fastvit_type not in supported_types:
        raise ValueError(f'Unknown FastViT Architecture: {fastvit_type}')

    try:
        return create_model(fastvit_type, pretrained=True, num_classes=num_classes)
    except RuntimeError as e:
        print(f"{fastvit_type} - Error loading pretrained model: {e}")
        return create_model(fastvit_type, pretrained=False, num_classes=num_classes)
