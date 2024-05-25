from torch import nn
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
    # Mapping of vision types to their corresponding torchvision models and
    # weights
    torchvision_models = {
        "MaxVit_T": (models.maxvit_t, models.MaxVit_T_Weights),
    }

    timm_models = [
        "coatnet_pico_rw_224",
        "coatnet_nano_rw_224",
        "coatnet_0_rw_224",
        "coatnet_1_rw_224",
        "coatnet_2_rw_224",
        "coatnet_3_rw_224",
        "coatnet_bn_0_rw_224",
        "coatnet_rmlp_nano_rw_224",
        "coatnet_rmlp_0_rw_224",
        "coatnet_rmlp_1_rw_224",
        "coatnet_rmlp_1_rw2_224",
        "coatnet_rmlp_2_rw_224",
        "coatnet_rmlp_2_rw_384",
        "coatnet_rmlp_3_rw_224",
        "coatnet_nano_cc_224",
        "coatnext_nano_rw_224",
        "coatnet_0_224",
        "coatnet_1_224",
        "coatnet_2_224",
        "coatnet_3_224",
        "coatnet_4_224",
        "coatnet_5_224",
        "maxvit_pico_rw_256",
        "maxvit_nano_rw_256",
        "maxvit_tiny_rw_224",
        "maxvit_tiny_rw_256",
        "maxvit_rmlp_pico_rw_256",
        "maxvit_rmlp_nano_rw_256",
        "maxvit_rmlp_tiny_rw_256",
        "maxvit_rmlp_small_rw_224",
        "maxvit_rmlp_small_rw_256",
        "maxvit_rmlp_base_rw_224",
        "maxvit_rmlp_base_rw_384",
        "maxvit_tiny_pm_256",
        "maxxvit_rmlp_nano_rw_256",
        "maxxvit_rmlp_tiny_rw_256",
        "maxxvit_rmlp_small_rw_256",
        "maxxvitv2_nano_rw_256",
        "maxxvitv2_rmlp_base_rw_224",
        "maxxvitv2_rmlp_base_rw_384",
        "maxxvitv2_rmlp_large_rw_224",
        "maxvit_tiny_tf_224",
        "maxvit_tiny_tf_384",
        "maxvit_tiny_tf_512",
        "maxvit_small_tf_224",
        "maxvit_small_tf_384",
        "maxvit_small_tf_512",
        "maxvit_base_tf_224",
        "maxvit_base_tf_384",
        "maxvit_base_tf_512",
        "maxvit_large_tf_224",
        "maxvit_large_tf_384",
        "maxvit_large_tf_512",
        "maxvit_xlarge_tf_224",
        "maxvit_xlarge_tf_384",
        "maxvit_xlarge_tf_512",
    ]

    # Check if the vision type is from torchvision
    if maxvit_type in torchvision_models:
        model_func, weights_class = torchvision_models[maxvit_type]
        try:
            weights = weights_class.DEFAULT
            maxvit_model = model_func(weights=weights)
        except RuntimeError as e:
            print(f"{maxvit_type} - Error loading pretrained model: {e}")
            maxvit_model = model_func(weights=None)

        # Modify the last layer to suit the given number of classes
        num_features = maxvit_model.classifier[-1].in_features
        maxvit_model.classifier[-1] = nn.Linear(num_features, num_classes)

    # Check if the vision type is from the 'timm' library
    elif maxvit_type in timm_models:
        try:
            maxvit_model = create_model(
                maxvit_type,
                pretrained=True,
                num_classes=num_classes,
            )
        except RuntimeError as e:
            print(f"{maxvit_type} - Error loading pretrained model: {e}")
            maxvit_model = create_model(
                maxvit_type,
                pretrained=False,
                num_classes=num_classes,
            )
    else:
        msg = f"Unknown MaxVit Architecture: {maxvit_type}"
        raise ValueError(msg)

    return maxvit_model
