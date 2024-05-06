import torch.nn as nn
from torchvision import models
from timm import create_model


def get_resnet_model(resnet_type, num_classes):
    """
    Load a ResNetV2 model based on the specified type.

    Parameters:
        resnet_type (str): The type of ResNet model to load. Options include:
            'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'resnet10t', 'resnet14t',
            'resnet18', 'resnet18d', 'resnet34', 'resnet34d', 'resnet26', 'resnet26t', 'resnet26d',
            'resnet50', 'resnet50c', 'resnet50d', 'resnet50s', 'resnet50t', 'resnet101', 'resnet101c',
            'resnet101d', 'resnet101s', 'resnet152', 'resnet152c', 'resnet152d', 'resnet152s', 'resnet200',
            'resnet200d', 'wide_resnet50_2', 'wide_resnet101_2', 'resnet50_gn', 'resnext50_32x4d',
            'resnext50d_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_32x16d',
            'resnext101_32x32d', 'resnext101_64x4d', 'ecaresnet26t', 'ecaresnet50d', 'ecaresnet50d_pruned',
            'ecaresnet50t', 'ecaresnetlight', 'ecaresnet101d', 'ecaresnet101d_pruned', 'ecaresnet200d',
            'ecaresnet269d', 'ecaresnext26t_32x4d', 'ecaresnext50t_32x4d', 'seresnet18', 'seresnet34',
            'seresnet50', 'seresnet50t', 'seresnet101', 'seresnet152', 'seresnet152d', 'seresnet200d',
            'seresnet269d', 'seresnext26d_32x4d', 'seresnext26t_32x4d', 'seresnext50_32x4d',
            'seresnext101_32x8d', 'seresnext101d_32x8d', 'seresnext101_64x4d', 'senet154', 'resnetblur18',
            'resnetblur50', 'resnetblur50d', 'resnetblur101d', 'resnetaa34d', 'resnetaa50', 'resnetaa50d',
            'resnetaa101d', 'seresnetaa50d', 'seresnextaa101d_32x8d', 'seresnextaa201d_32x8d', 'resnetrs50',
            'resnetrs101', 'resnetrs152', 'resnetrs200', 'resnetrs270', 'resnetrs350', 'resnetrs420'
        num_classes (int): The number of output classes for the model.

    Returns:
        torch.nn.Module: The loaded ResNetV2 model.

    Raises:
        ValueError: If the specified ResNetV2 architecture is unknown.
    """
    torch_vision = False
    if resnet_type == 'ResNet18':
        torch_vision = True
        try:
            weights = models.ResNet18_Weights.DEFAULT
            resnet_model = models.resnet18(weights=weights)
        except:
            resnet_model = models.resnet18(weights=None)
    elif resnet_type == 'ResNet34':
        torch_vision = True
        try:
            weights = models.ResNet34_Weights.DEFAULT
            resnet_model = models.resnet34(weights=weights)
        except:
            resnet_model = models.resnet34(weights=None)
    elif resnet_type == 'ResNet50':
        torch_vision = True
        try:
            weights = models.ResNet50_Weights.DEFAULT
            resnet_model = models.resnet50(weights=weights)
        except:
            resnet_model = models.resnet50(weights=None)
    elif resnet_type == 'ResNet101':
        torch_vision = True
        try:
            weights = models.ResNet101_Weights.DEFAULT
            resnet_model = models.resnet101(weights=weights)
        except:
            resnet_model = models.resnet101(weights=None)
    elif resnet_type == 'ResNet152':
        torch_vision = True
        try:
            weights = models.ResNet152_Weights.DEFAULT
            resnet_model = models.resnet152(weights=weights)
        except:
            resnet_model = models.resnet152(weights=None)
    elif resnet_type == 'resnet10t':
        try:
            resnet_model = create_model('resnet10t',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet10t',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet14t':
        try:
            resnet_model = create_model('resnet14t',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet14t',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet18':
        try:
            resnet_model = create_model('resnet18',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet18',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet18d':
        try:
            resnet_model = create_model('resnet18d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet18d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet34':
        try:
            resnet_model = create_model('resnet34',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet34',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet34d':
        try:
            resnet_model = create_model('resnet34d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet34d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet26':
        try:
            resnet_model = create_model('resnet26',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet26',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet26t':
        try:
            resnet_model = create_model('resnet26t',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet26t',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet26d':
        try:
            resnet_model = create_model('resnet26d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet26d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet50':
        try:
            resnet_model = create_model('resnet50',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet50',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet50c':
        try:
            resnet_model = create_model('resnet50c',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet50c',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet50d':
        try:
            resnet_model = create_model('resnet50d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet50d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet50s':
        try:
            resnet_model = create_model('resnet50s',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet50s',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet50t':
        try:
            resnet_model = create_model('resnet50t',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet50t',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet101':
        try:
            resnet_model = create_model('resnet101',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet101',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet101c':
        try:
            resnet_model = create_model('resnet101c',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet101c',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet101d':
        try:
            resnet_model = create_model('resnet101d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet101d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet101s':
        try:
            resnet_model = create_model('resnet101s',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet101s',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet152':
        try:
            resnet_model = create_model('resnet152',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet152',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet152c':
        try:
            resnet_model = create_model('resnet152c',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet152c',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet152d':
        try:
            resnet_model = create_model('resnet152d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet152d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet152s':
        try:
            resnet_model = create_model('resnet152s',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet152s',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet200':
        try:
            resnet_model = create_model('resnet200',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet200',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet200d':
        try:
            resnet_model = create_model('resnet200d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet200d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'wide_resnet50_2':
        try:
            resnet_model = create_model('wide_resnet50_2',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('wide_resnet50_2',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'wide_resnet101_2':
        try:
            resnet_model = create_model('wide_resnet101_2',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('wide_resnet101_2',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnet50_gn':
        try:
            resnet_model = create_model('resnet50_gn',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnet50_gn',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnext50_32x4d':
        try:
            resnet_model = create_model('resnext50_32x4d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnext50_32x4d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnext50d_32x4d':
        try:
            resnet_model = create_model('resnext50d_32x4d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnext50d_32x4d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnext101_32x4d':
        try:
            resnet_model = create_model('resnext101_32x4d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnext101_32x4d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnext101_32x8d':
        try:
            resnet_model = create_model('resnext101_32x8d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnext101_32x8d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnext101_32x16d':
        try:
            resnet_model = create_model('resnext101_32x16d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnext101_32x16d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnext101_32x32d':
        try:
            resnet_model = create_model('resnext101_32x32d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnext101_32x32d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnext101_64x4d':
        try:
            resnet_model = create_model('resnext101_64x4d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnext101_64x4d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'ecaresnet26t':
        try:
            resnet_model = create_model('ecaresnet26t',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('ecaresnet26t',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'ecaresnet50d':
        try:
            resnet_model = create_model('ecaresnet50d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('ecaresnet50d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'ecaresnet50d_pruned':
        try:
            resnet_model = create_model('ecaresnet50d_pruned',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('ecaresnet50d_pruned',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'ecaresnet50t':
        try:
            resnet_model = create_model('ecaresnet50t',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('ecaresnet50t',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'ecaresnetlight':
        try:
            resnet_model = create_model('ecaresnetlight',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('ecaresnetlight',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'ecaresnet101d':
        try:
            resnet_model = create_model('ecaresnet101d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('ecaresnet101d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'ecaresnet101d_pruned':
        try:
            resnet_model = create_model('ecaresnet101d_pruned',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('ecaresnet101d_pruned',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'ecaresnet200d':
        try:
            resnet_model = create_model('ecaresnet200d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('ecaresnet200d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'ecaresnet269d':
        try:
            resnet_model = create_model('ecaresnet269d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('ecaresnet269d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'ecaresnext26t_32x4d':
        try:
            resnet_model = create_model('ecaresnext26t_32x4d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('ecaresnext26t_32x4d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'ecaresnext50t_32x4d':
        try:
            resnet_model = create_model('ecaresnext50t_32x4d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('ecaresnext50t_32x4d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnet18':
        try:
            resnet_model = create_model('seresnet18',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnet18',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnet34':
        try:
            resnet_model = create_model('seresnet34',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnet34',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnet50':
        try:
            resnet_model = create_model('seresnet50',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnet50',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnet50t':
        try:
            resnet_model = create_model('seresnet50t',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnet50t',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnet101':
        try:
            resnet_model = create_model('seresnet101',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnet101',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnet152':
        try:
            resnet_model = create_model('seresnet152',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnet152',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnet152d':
        try:
            resnet_model = create_model('seresnet152d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnet152d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnet200d':
        try:
            resnet_model = create_model('seresnet200d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnet200d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnet269d':
        try:
            resnet_model = create_model('seresnet269d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnet269d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnext26d_32x4d':
        try:
            resnet_model = create_model('seresnext26d_32x4d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnext26d_32x4d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnext26t_32x4d':
        try:
            resnet_model = create_model('seresnext26t_32x4d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnext26t_32x4d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnext50_32x4d':
        try:
            resnet_model = create_model('seresnext50_32x4d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnext50_32x4d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnext101_32x4d':
        try:
            resnet_model = create_model('seresnext101_32x4d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnext101_32x4d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnext101_32x8d':
        try:
            resnet_model = create_model('seresnext101_32x8d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnext101_32x8d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnext101d_32x8d':
        try:
            resnet_model = create_model('seresnext101d_32x8d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnext101d_32x8d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnext101_64x4d':
        try:
            resnet_model = create_model('seresnext101_64x4d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnext101_64x4d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'senet154':
        try:
            resnet_model = create_model('senet154',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('senet154',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnetblur18':
        try:
            resnet_model = create_model('resnetblur18',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnetblur18',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnetblur50':
        try:
            resnet_model = create_model('resnetblur50',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnetblur50',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnetblur50d':
        try:
            resnet_model = create_model('resnetblur50d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnetblur50d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnetblur101d':
        try:
            resnet_model = create_model('resnetblur101d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnetblur101d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnetaa34d':
        try:
            resnet_model = create_model('resnetaa34d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnetaa34d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnetaa50':
        try:
            resnet_model = create_model('resnetaa50',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnetaa50',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnetaa50d':
        try:
            resnet_model = create_model('resnetaa50d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnetaa50d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnetaa101d':
        try:
            resnet_model = create_model('resnetaa101d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnetaa101d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnetaa50d':
        try:
            resnet_model = create_model('seresnetaa50d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnetaa50d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnextaa101d_32x8d':
        try:
            resnet_model = create_model('seresnextaa101d_32x8d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnextaa101d_32x8d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'seresnextaa201d_32x8d':
        try:
            resnet_model = create_model('seresnextaa201d_32x8d',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('seresnextaa201d_32x8d',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnetrs50':
        try:
            resnet_model = create_model('resnetrs50',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnetrs50',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnetrs101':
        try:
            resnet_model = create_model('resnetrs101',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnetrs101',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnetrs152':
        try:
            resnet_model = create_model('resnetrs152',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnetrs152',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnetrs200':
        try:
            resnet_model = create_model('resnetrs200',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnetrs200',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnetrs270':
        try:
            resnet_model = create_model('resnetrs270',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnetrs270',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnetrs350':
        try:
            resnet_model = create_model('resnetrs350',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnetrs350',
                                        pretrained=False,
                                        num_classes=num_classes)
    elif resnet_type == 'resnetrs420':
        try:
            resnet_model = create_model('resnetrs420',
                                        pretrained=True,
                                        num_classes=num_classes)
        except:
            resnet_model = create_model('resnetrs420',
                                        pretrained=False,
                                        num_classes=num_classes)
    else:
        raise ValueError(f'Unknown ResNet Architecture: {resnet_type}')

    if torch_vision:
        # Modify the last layer to suit the given number of classes
        num_features = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(num_features, num_classes)

    return resnet_model
