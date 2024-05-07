import torch.nn as nn
from torchvision import models
from timm import create_model


def get_maxvit_model(maxvit_type, num_classes):
    """
    Returns a modified MaxVit model based on the specified type.

    Parameters:
        - maxvit_type (str): Type of MaxVit architecture.
                            Currently supports 'MaxVit_T'.
        - num_classes (int): Number of classes for the modified last layer.

    Returns:
        - torch.nn.Module: Modified MaxVit model with the specified number of classes.

    Raises:
        - ValueError: If an unknown MaxVit architecture type is provided.
    """
    torch_vision = False
    # Load the pre-trained version of MaxVit based on the specified type
    if maxvit_type == 'MaxVit_T':
        torch_vision = True
        try:
            weights = models.MaxVit_T_Weights.DEFAULT
            maxvit_model = models.maxvit_t(weights=weights)
        except ValueError:
            maxvit_model = models.maxvit_t(weights=None)
    elif maxvit_type == "coatnet_pico_rw_224":
        try:
            maxvit_model = create_model('coatnet_pico_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_pico_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_nano_rw_224":
        try:
            maxvit_model = create_model('coatnet_nano_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_nano_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_0_rw_224":
        try:
            maxvit_model = create_model('coatnet_0_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_0_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_1_rw_224":
        try:
            maxvit_model = create_model('coatnet_1_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_1_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_2_rw_224":
        try:
            maxvit_model = create_model('coatnet_2_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_2_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_3_rw_224":
        try:
            maxvit_model = create_model('coatnet_3_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_3_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_bn_0_rw_224":
        try:
            maxvit_model = create_model('coatnet_bn_0_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_bn_0_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_rmlp_nano_rw_224":
        try:
            maxvit_model = create_model('coatnet_rmlp_nano_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_rmlp_nano_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_rmlp_0_rw_224":
        try:
            maxvit_model = create_model('coatnet_rmlp_0_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_rmlp_0_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_rmlp_1_rw_224":
        try:
            maxvit_model = create_model('coatnet_rmlp_1_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_rmlp_1_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_rmlp_1_rw2_224":
        try:
            maxvit_model = create_model('coatnet_rmlp_1_rw2_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_rmlp_1_rw2_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_rmlp_2_rw_224":
        try:
            maxvit_model = create_model('coatnet_rmlp_2_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_rmlp_2_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_rmlp_2_rw_384":
        try:
            maxvit_model = create_model('coatnet_rmlp_2_rw_384',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_rmlp_2_rw_384',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_rmlp_3_rw_224":
        try:
            maxvit_model = create_model('coatnet_rmlp_3_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_rmlp_3_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_nano_cc_224":
        try:
            maxvit_model = create_model('coatnet_nano_cc_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_nano_cc_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnext_nano_rw_224":
        try:
            maxvit_model = create_model('coatnext_nano_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnext_nano_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_0_224":
        try:
            maxvit_model = create_model('coatnet_0_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_0_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_1_224":
        try:
            maxvit_model = create_model('coatnet_1_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_1_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_2_224":
        try:
            maxvit_model = create_model('coatnet_2_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_2_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_3_224":
        try:
            maxvit_model = create_model('coatnet_3_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_3_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_4_224":
        try:
            maxvit_model = create_model('coatnet_4_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_4_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "coatnet_5_224":
        try:
            maxvit_model = create_model('coatnet_5_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('coatnet_5_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_pico_rw_256":
        try:
            maxvit_model = create_model('maxvit_pico_rw_256',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_pico_rw_256',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_nano_rw_256":
        try:
            maxvit_model = create_model('maxvit_nano_rw_256',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_nano_rw_256',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_tiny_rw_224":
        try:
            maxvit_model = create_model('maxvit_tiny_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_tiny_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_tiny_rw_256":
        try:
            maxvit_model = create_model('maxvit_tiny_rw_256',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_tiny_rw_256',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_rmlp_pico_rw_256":
        try:
            maxvit_model = create_model('maxvit_rmlp_pico_rw_256',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_rmlp_pico_rw_256',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_rmlp_nano_rw_256":
        try:
            maxvit_model = create_model('maxvit_rmlp_nano_rw_256',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_rmlp_nano_rw_256',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_rmlp_tiny_rw_256":
        try:
            maxvit_model = create_model('maxvit_rmlp_tiny_rw_256',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_rmlp_tiny_rw_256',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_rmlp_small_rw_224":
        try:
            maxvit_model = create_model('maxvit_rmlp_small_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_rmlp_small_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_rmlp_small_rw_256":
        try:
            maxvit_model = create_model('maxvit_rmlp_small_rw_256',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_rmlp_small_rw_256',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_rmlp_base_rw_224":
        try:
            maxvit_model = create_model('maxvit_rmlp_base_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_rmlp_base_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_rmlp_base_rw_384":
        try:
            maxvit_model = create_model('maxvit_rmlp_base_rw_384',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_rmlp_base_rw_384',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_tiny_pm_256":
        try:
            maxvit_model = create_model('maxvit_tiny_pm_256',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_tiny_pm_256',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxxvit_rmlp_nano_rw_256":
        try:
            maxvit_model = create_model('maxxvit_rmlp_nano_rw_256',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxxvit_rmlp_nano_rw_256',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxxvit_rmlp_tiny_rw_256":
        try:
            maxvit_model = create_model('maxxvit_rmlp_tiny_rw_256',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxxvit_rmlp_tiny_rw_256',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxxvit_rmlp_small_rw_256":
        try:
            maxvit_model = create_model('maxxvit_rmlp_small_rw_256',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxxvit_rmlp_small_rw_256',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxxvitv2_nano_rw_256":
        try:
            maxvit_model = create_model('maxxvitv2_nano_rw_256',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxxvitv2_nano_rw_256',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxxvitv2_rmlp_base_rw_224":
        try:
            maxvit_model = create_model('maxxvitv2_rmlp_base_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxxvitv2_rmlp_base_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxxvitv2_rmlp_base_rw_384":
        try:
            maxvit_model = create_model('maxxvitv2_rmlp_base_rw_384',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxxvitv2_rmlp_base_rw_384',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxxvitv2_rmlp_large_rw_224":
        try:
            maxvit_model = create_model('maxxvitv2_rmlp_large_rw_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxxvitv2_rmlp_large_rw_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_tiny_tf_224":
        try:
            maxvit_model = create_model('maxvit_tiny_tf_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_tiny_tf_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_tiny_tf_384":
        try:
            maxvit_model = create_model('maxvit_tiny_tf_384',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_tiny_tf_384',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_tiny_tf_512":
        try:
            maxvit_model = create_model('maxvit_tiny_tf_512',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_tiny_tf_512',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_small_tf_224":
        try:
            maxvit_model = create_model('maxvit_small_tf_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_small_tf_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_small_tf_384":
        try:
            maxvit_model = create_model('maxvit_small_tf_384',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_small_tf_384',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_small_tf_512":
        try:
            maxvit_model = create_model('maxvit_small_tf_512',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_small_tf_512',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_base_tf_224":
        try:
            maxvit_model = create_model('maxvit_base_tf_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_base_tf_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_base_tf_384":
        try:
            maxvit_model = create_model('maxvit_base_tf_384',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_base_tf_384',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_base_tf_512":
        try:
            maxvit_model = create_model('maxvit_base_tf_512',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_base_tf_512',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_large_tf_224":
        try:
            maxvit_model = create_model('maxvit_large_tf_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_large_tf_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_large_tf_384":
        try:
            maxvit_model = create_model('maxvit_large_tf_384',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_large_tf_384',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_large_tf_512":
        try:
            maxvit_model = create_model('maxvit_large_tf_512',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_large_tf_512',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_xlarge_tf_224":
        try:
            maxvit_model = create_model('maxvit_xlarge_tf_224',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_xlarge_tf_224',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_xlarge_tf_384":
        try:
            maxvit_model = create_model('maxvit_xlarge_tf_384',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_xlarge_tf_384',
                                       pretrained=False,
                                       num_classes=num_classes)
    elif maxvit_type == "maxvit_xlarge_tf_512":
        try:
            maxvit_model = create_model('maxvit_xlarge_tf_512',
                                       pretrained=True,
                                       num_classes=num_classes)
        except ValueError:
            maxvit_model = create_model('maxvit_xlarge_tf_512',
                                       pretrained=False,
                                       num_classes=num_classes)
    else:
        raise ValueError(f'Unknown MaxVit Architecture: {maxvit_type}')

    if torch_vision:
        # Modify the last layer to suit the given number of classes
        num_features = maxvit_model.classifier[-1].in_features
        maxvit_model.classifier[-1] = nn.Linear(num_features, num_classes)

    return maxvit_model
