import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.cspnet import get_cspnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingCspnetTestCase(TestCase):
    """
    Test case class for processing Cspnet models.
    """

    def test_known_cspnet_types(self):
        """
        Test if the function returns a valid CSPNet model for known CSPNet types.
        """
        cspnet_types = [
            'cspresnet50', 'cspresnet50d', 'cspresnet50w', 'cspresnext50', 'cspdarknet53', 'darknet17', 'darknet21',
            'sedarknet21', 'darknet53', 'darknetaa53', 'cs3darknet_s', 'cs3darknet_m', 'cs3darknet_l', 'cs3darknet_x',
            'cs3darknet_focus_s', 'cs3darknet_focus_m', 'cs3darknet_focus_l', 'cs3darknet_focus_x', 'cs3sedarknet_l',
            'cs3sedarknet_x', 'cs3sedarknet_xdw', 'cs3edgenet_x', 'cs3se_edgenet_x'
        ]
        num_classes = 10

        for cspnet_type in cspnet_types:
            cspnet_model = get_cspnet_model(cspnet_type, num_classes)
            self.assertIsNotNone(cspnet_model)
            # Check if the model has the expected number of output classes
            self.assertEqual(cspnet_model.num_classes, num_classes)

    def test_unknown_cspnet_type(self):
        """
        Test if the function raises a ValueError for an unknown CSPNet type.
        """
        unknown_cspnet_type = 'unknown_cspnet_type'
        num_classes = 10

        with self.assertRaises(ValueError):
            get_cspnet_model(unknown_cspnet_type, num_classes)
