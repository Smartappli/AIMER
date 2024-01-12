from django.test import TestCase
from fl_common.models.vgg import get_vgg_model
from fl_common.models.swin import get_swin_model
from fl_common.models.shufflenet import get_shufflenet_model
class ProcessinfTestCase(TestCase):
    def test_get_vgg_model(self):
        vgg11 = get_vgg_model('VGG11',1000)
        self.assertIsNotNone(vgg11, msg="get_vgg_model KO")

    def test_get_swim_model(self):
        swin = get_swin_model('Swin_T', 1000)
        self.assertIsNotNone(swin, msg="get_swin_model KO")

    def test_get_shufflenet_model(self):
        shufflenet = get_shufflenet_model('ShuffleNet_V2_X0_5', 1000)
        self.assertIsNotNone(shufflenet, msg="get_shufflenet_model KO")