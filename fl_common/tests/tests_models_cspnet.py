import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.cspnet import get_cspnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartCTestCase(TestCase):
    # Cspnet model unit tests
    def test_get_cspnet_model(self):
        """
        Test the get_cspnet_model function.

        This test iterates over various configurations of CSPNet models
        and ensures that the returned models are not None and instances
        of nn.Module.

        Parameters:
            None

        Returns:
            None
        """
        num_classes = 1000  # Example number of classes
        cspnet_types = [
            "cspresnet50", "cspresnet50d", "cspresnet50w", "cspresnext50",
            "cspdarknet53", "darknet17", "darknet21", "sedarknet21",
            "darknet53", "darknetaa53", "cs3darknet_s", "cs3darknet_m",
            "cs3darknet_l", "cs3darknet_x", "cs3darknet_focus_s", "cs3darknet_focus_m",
            "cs3darknet_focus_l", "cs3darknet_focus_x", "cs3sedarknet_l", "cs3sedarknet_x",
            "cs3sedarknet_xdw", "cs3edgenet_x", "cs3se_edgenet_x"
        ]

        for cspnet_type in cspnet_types:
            with self.subTest(cspnet_type=cspnet_type):
                model = get_cspnet_model(cspnet_type, num_classes)
                self.assertIsNotNone(model)
                self.assertIsInstance(model, nn.Module)
