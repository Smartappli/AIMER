from django.test import TestCase
from fl_common.models.utils import (get_optimizer,
                                    get_criterion,
                                    get_scheduler,
                                    generate_xai_heatmaps,
                                    get_dataset,
                                    EarlyStopping)
from fl_common.models.alexnet import get_alexnet_model
from fl_common.models.convnext import get_convnext_model
from fl_common.models.densenet import get_densenet_model
from fl_common.models.efficientnet import get_efficientnet_model
from fl_common.models.googlenet import get_googlenet_model
from fl_common.models.inception import get_inception_model
from fl_common.models.maxvit import get_maxvit_model
from fl_common.models.mnasnet import get_mnasnet_model
from fl_common.models.mobilenet import get_mobilenet_model
from fl_common.models.regnet import get_regnet_model
from fl_common.models.resnet import get_resnet_model
from fl_common.models.resnext import get_resnext_model
from fl_common.models.shufflenet import get_shufflenet_model
from fl_common.models.squeezenet import get_squeezenet_model
from fl_common.models.swin_transformer import get_swin_model
from fl_common.models.vgg import get_vgg_model
from fl_common.models.vision_transformer import get_vision_model
from fl_common.models.wide_resnet import get_wide_resnet_model


class ProcessingTestCase(TestCase):
    def test_get_wide_resnet_model(self):
        wide_resnet_model = get_wide_resnet_model('Wide_ResNet50_2', 1000)
        self.assertIsNotNone(wide_resnet_model, msg="Wide ResNet KO")

    def test_get_vision_model(self):
        vision_model = get_vision_model('ViT_B_16', 1000)
        self.assertIsNotNone(vision_model, msg="Wision Transform KO")

    def test_get_vgg_model(self):
        vgg11 = get_vgg_model('VGG11',1000)
        self.assertIsNotNone(vgg11, msg="get_vgg_model KO")

    def test_get_swim_model(self):
        swin = get_swin_model('Swin_T', 1000)
        self.assertIsNotNone(swin, msg="get_swin_model KO")

    def test_get_squeezenet_model(self):
        squeezenet_model = get_squeezenet_model('SqueezeNet1_0', 1000)
        self.assertIsNotNone(squeezenet_model, msg="get_squeezenet_model KO")

    def test_get_shufflenet_model(self):
        shufflenet = get_shufflenet_model('ShuffleNet_V2_X0_5', 1000)
        self.assertIsNotNone(shufflenet, msg="get_shufflenet_model KO")

    def test_get_resnext_model(self):
        resnext_model = get_resnext_model('ResNeXt50_32X4D', 1000)
        self.assertIsNotNone(resnext_model, msg="get_resnext_model KO")

    def test_get_resnet_model(self):
        resnet = get_resnet_model('ResNet50', 1000)
        self.assertIsNotNone(resnet, msg="get_resnet_model KO")

    def test_get_regnet_model(self):
        regnet = get_regnet_model('RegNet_X_400MF', 1000)
        self.assertIsNotNone(regnet,  msg="get_regnet_model KO")

    def test_get_mobilenet_model(self):
        mobilenet = get_mobilenet_model('MobileNet_V3_Small', 1000)
        self.assertIsNotNone(mobilenet, msg="get_mobilenet_model KO")

    def test_get_mnasnet_model(self):
        mnasnet = get_mnasnet_model('MNASNet0_5', 1000)
        self.assertIsNotNone(mnasnet, msg="get_mnasnet_model KO")

    def test_get_maxvit_model(self):
        maxvit = get_maxvit_model('MaxVit_T', 1000)
        self.assertIsNotNone(maxvit, msg="get_maxvit_model KO")

    def test_get_inception_model(self):
        inception = get_inception_model('Inception_V3', 1000)
        self.assertIsNotNone(inception, msg="get_inception_model KO")

    def test_get_googlenet_model(self):
        googlenet = get_googlenet_model('GoogLeNet', 1000)
        self.assertIsNotNone(googlenet, msg="get_googlenet_model KO")

    def test_efficientnet_model(self):
        efficientnet = get_efficientnet_model('EfficientNetB0', 1000)
        self.assertIsNotNone(efficientnet, msg="get_efficientnet_model KO")

    def test_densenet_model(self):
        densenet = get_densenet_model('DenseNet121', 1000)
        self.assertIsNotNone(densenet, msg="get_densenet_model KO")

    def test_convnet_model(self):
        convnext = get_convnext_model('ConvNeXt_Tiny', 1000)
        self.assertIsNotNone(convnext, msg="get_convnext_model KO")

    def test_alexnet_model(self):
        alexnet = get_alexnet_model('AlexNet', 1000)
        self.assertIsNotNone(alexnet, msg="get_alexnet_model KO")
