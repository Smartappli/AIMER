from fl_common.models.convnext.convnext import get_convnext_model
from fl_common.models.densenet.densenet import get_densenet_model
from fl_common.models.efficientnet.efficientnet import get_efficientnet_model
from fl_common.models.resnet.resnet import get_resnet_model
from fl_common.models.swin_transformer.swin import get_swin_model
from fl_common.models.vgg.vgg import get_vgg_model

model_familly = 'vgg'


def get_familly_model(model_type):
    """
    Returns a model based on the specified model family and type.

    Parameters:
    - model_type (str): Type of the model within the specified family.

    Returns:
    - model (torch.nn.Module): Model corresponding to the specified family and type.

    Raises:
    - ValueError: If the specified model family is not recognized.
    """

    match model_familly:
        case 'vgg':
            return get_vgg_model(model_type)

        case 'resnet':
            return get_resnet_model(model_type)

        case 'densenet':
            return get_densenet_model(model_type)

        case 'swin':
            return get_swin_model(model_type)

        #case _:
        #    nothing_matched_function()
