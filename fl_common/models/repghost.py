from timm import create_model


def get_repghost_model(repghost_type, num_classes):
    """
    Returns a Residual-Path Ghost Network (RepGhost) model based on the provided RepGhost type and number of classes.

    Args:
    - repghost_type (str): The type of RepGhost model. It should be one of the following:
        - 'repghostnet_050'
        - 'repghostnet_058'
        - 'repghostnet_080'
        - 'repghostnet_100'
        - 'repghostnet_111'
        - 'repghostnet_130'
        - 'repghostnet_150'
        - 'repghostnet_200'
    - num_classes (int): The number of output classes for the model.

    Returns:
    - repghost_model: The RepGhost model instantiated based on the specified architecture.

    Raises:
    - ValueError: If the provided repghost_type is not recognized.
    """
    if repghost_type == 'repghostnet_050':
        repghost_model = create_model('repghostnet_050',pretrained=True, num_classes=num_classes)
    elif repghost_type == 'repghostnet_058':
        repghost_model = create_model('repghostnet_058',pretrained=True, num_classes=num_classes)
    elif repghost_type == 'repghostnet_080':
        repghost_model = create_model('repghostnet_080',pretrained=True, num_classes=num_classes)
    elif repghost_type == 'repghostnet_100':
        repghost_model = create_model('repghostnet_100',pretrained=True, num_classes=num_classes)
    elif repghost_type == 'repghostnet_111':
        repghost_model = create_model('repghostnet_111',pretrained=True,num_classes=num_classes)
    elif repghost_type == 'repghostnet_130':
        repghost_model = create_model('repghostnet_130', pretrained=True, num_classes=num_classes)
    elif repghost_type == 'repghostnet_150':
        repghost_model = create_model('repghostnet_150',pretrained=True, num_classes=num_classes)
    elif repghost_type == 'repghostnet_200':
        repghost_model = create_model('repghostnet_200',pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown Repghost Architecture: {repghost_type}')

    return repghost_model
