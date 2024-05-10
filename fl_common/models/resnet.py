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
    # Mapping of vision types to their corresponding torchvision models and weights
    torchvision_models = {
        'ResNet18': (models.resnet18, models.ResNet18_Weights),
        'ResNet34': (models.resnet_small, models.ResNet34_Weights),
        'ResNet50': (models.resnet50, models.ResNet50_Weights),
        'ResNet101': (models.resnet101, models.ResNet101_Weights),
        'ResNet152': (models.resnet152, models.ResNet152_Weights)        
    }

    timm_models = [
        "resnet10t", "resnet14t", "resnet18", "resnet18d", "resnet34",
        "resnet34d", "resnet26", "resnet26t", "resnet26d", "resnet50",
        "resnet50c", "resnet50d", "resnet50s", "resnet50t", "resnet101",
        "resnet101c", "resnet101d", "resnet101s", "resnet152", "resnet152c",
        "resnet152d", "resnet152s", "resnet200", "resnet200d", "wide_resnet50_2",
        "resnet50_gn", "resnext50d_32x4d", "resnext101_32x4d", "resnext101_32x8d",
        "resnext101_32x16d", "resnext101_32x32d", "resnext101_32x32d",
        "resnext101_64x4d", "ecaresnet26t", "ecaresnet50d", "ecaresnet50d_pruned",
        "ecaresnet50t", "ecaresnetlight", "ecaresnet101d", "ecaresnet101d_pruned",
        "ecaresnet200d", "ecaresnet269d", "ecaresnext26t_32x4d", "ecaresnext50t_32x4d",
        "seresnet18", "seresnet34", "seresnet50", "seresnet50t", "seresnet101",
        "seresnet152", "seresnet152d", "seresnet200d", "seresnet269d",
        "seresnext26d_32x4d", "seresnext26t_32x4d", "seresnext50_32x4d",
        "seresnext101_32x4d", "seresnext101_32x8d", "seresnext101d_32x8d",
        "seresnext101_64x4d", "senet154", "resnetblur18", "resnetblur50",
        "resnetblur50d", "resnetblur101d", "resnetaa34d", "resnetaa50", "resnetaa50d",
        "resnetaa101d", "seresnetaa50d", "seresnextaa101d_32x8d", "seresnextaa201d_32x8d",
        "resnetrs50", "resnetrs101", "resnetrs152", "resnetrs270", "resnetrs350",
        "resnetrs420"
    ]
    
    # Check if the vision type is from torchvision
    if resnet_type in torchvision_models:
        model_func, weights_class = torchvision_models[resnet_type]
        try:
            weights = weights_class.DEFAULT
            resnet_model = model_func(weights=weights)
        except RuntimeError as e:
            print(f"{resnet_type} - Error loading pretrained model: {e}")
            resnet_model = model_func(weights=None)

        # Modify the last layer to suit the given number of classes
        num_features = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(num_features, num_classes)

    # Check if the vision type is from the 'timm' library
    elif resnet_type in timm_models:
        try:
            resnet_model = create_model(resnet_type, pretrained=True, num_classes=num_classes)
        except RuntimeError as e:
            print(f"{resnet_type} - Error loading pretrained model: {e}")
            resnet_model = create_model(resnet_type, pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f'Unknown ResNet Architecture: {resnet_type}')

    return resnet_model
