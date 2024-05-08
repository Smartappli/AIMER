import torch.nn as nn
from torchvision import models
from timm import create_model


def get_convnext_model(convnext_type, num_classes):
    """
    Obtain a ConvNeXt model with a specified architecture type and modify it for the given number of classes.

    Args:
    - convnext_type (str): Type of ConvNeXt architecture to be loaded. Options: 'ConvNeXt_Tiny', 'ConvNeXt_Small',
      'ConvNeXt_Base', 'ConvNeXt_Large'. Default is 'ConvNeXt_Large'.
    - num_classes (int): Number of output classes for the modified model. Default is 1000.

    Returns:
    - convnext_model (torch.nn.Module): The modified ConvNeXt model.

    Raises:
    - ValueError: If the provided convnext_type is not recognized or the model structure is not as expected.

    Note:
    - This function loads a pre-trained ConvNeXt model and modifies its last fully connected layer to
      match the specified number of output classes.

    Example Usage:
    ```python
    # Obtain a ConvNeXt model with 10 output classes
    model = get_convnext_model(convnext_type='ConvNeXt_Small', num_classes=10)
    ```
    """
    torch_vision = False
    # Load the pre-trained version of DenseNet
    if convnext_type == 'ConvNeXt_Tiny':
        torch_vision = True
        try:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT
            convnext_model = models.convnext_tiny(weights=weights)
        except RuntimeError:
            convnext_model = models.convnext_tiny(weghts=None)
    elif convnext_type == 'ConvNeXt_Small':
        torch_vision = True
        try:
            weights = models.ConvNeXt_Small_Weights.DEFAULT
            convnext_model = models.convnext_small(weights=weights)
        except RuntimeError:
            convnext_model = models.convnext_small(weghts=None)
    elif convnext_type == 'ConvNeXt_Base':
        torch_vision = True
        try:
            weights = models.ConvNeXt_Base_Weights.DEFAULT
            convnext_model = models.convnext_base(weights=weights)
        except RuntimeError:
            convnext_model = models.convnext_base(weghts=None)
    elif convnext_type == 'ConvNeXt_Large':
        torch_vision = True
        try:
            weights = models.ConvNeXt_Large_Weights.DEFAULT
            convnext_model = models.convnext_large(weights=weights)
        except RuntimeError:
            convnext_model = models.convnext_large(weight=None)
    elif convnext_type == "convnext_atto":
        try:
            convnext_model = create_model('convnext_atto',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnext_atto',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnext_atto_ols":
        try:
            convnext_model = create_model('convnext_atto_ols',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnext_atto_ols',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnext_femto":
        try:
            convnext_model = create_model('convnext_femto',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnext_femto',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnext_femto_ols":
        try:
            convnext_model = create_model('convnext_femto_ols',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnext_femto_ols',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnext_pico":
        try:
            convnext_model = create_model('convnext_pico',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnext_pico',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnext_pico_ols":
        try:
            convnext_model = create_model('convnext_pico_ols',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnext_pico_ols',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnext_nano":
        try:
            convnext_model = create_model('convnext_nano',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnext_nano',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnext_nano_ols":
        try:
            convnext_model = create_model('convnext_nano_ols',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnext_nano_ols',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnext_tiny_hnf":
        try:
            convnext_model = create_model('convnext_tiny_hnf',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnext_tiny_hnf',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnext_tiny":
        try:
            convnext_model = create_model('convnext_tiny',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnext_tiny',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnext_small":
        try:
            convnext_model = create_model('convnext_small',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnext_small',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnext_base":
        try:
            convnext_model = create_model('convnext_base',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnext_base',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnext_large":
        try:
            convnext_model = create_model('convnext_large',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnext_large',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnext_large_mlp":
        try:
            convnext_model = create_model('convnext_large_mlp',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnext_large_mlp',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnext_xlarge":
        try:
            convnext_model = create_model('convnext_xlarge',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnext_xlarge',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnext_xxlarge":
        try:
            convnext_model = create_model('convnext_xxlarge',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnext_xxlarge',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnextv2_atto":
        try:
            convnext_model = create_model('convnextv2_atto',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnextv2_atto',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnextv2_femto":
        try:
            convnext_model = create_model('convnextv2_femto',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnextv2_femto',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnextv2_pico":
        try:
            convnext_model = create_model('convnextv2_pico',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnextv2_pico',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnextv2_nano":
        try:
            convnext_model = create_model('convnextv2_nano',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnextv2_nano',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnextv2_tiny":
        try:
            convnext_model = create_model('convnextv2_tiny',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnextv2_tiny',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnextv2_small":
        try:
            convnext_model = create_model('convnextv2_small',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnextv2_small',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnextv2_base":
        try:
            convnext_model = create_model('convnextv2_base',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnextv2_base',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnextv2_large":
        try:
            convnext_model = create_model('convnextv2_large',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnextv2_large',
                                          pretrained=False,
                                          num_classes=num_classes)
    elif convnext_type == "convnextv2_huge":
        try:
            convnext_model = create_model('convnextv2_huge',
                                          pretrained=True,
                                          num_classes=num_classes)
        except RuntimeError:
            convnext_model = create_model('convnextv2_huge',
                                          pretrained=False,
                                          num_classes=num_classes)
    else:
        raise ValueError(f'Unknown ConvNeXt Architecture : {convnext_type}')

    if torch_vision:
        # Modify last layer to suit number of classes
        for layer in reversed(convnext_model.classifier):
            if isinstance(layer, nn.Linear):
                # num_features = layer.in_features
                layer.out_features = num_classes
                break

    return convnext_model
