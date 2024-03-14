import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.crossvit import get_crossvit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingCrossvitTestCase(TestCase):
    """
    Test case class for processing Crossvit models.
    """

    def test_get_crossvit_model(self):
        """
        Test the get_crossvit_model function.

        This test iterates over various configurations of CrossViT models
        and ensures that the returned models are not None and instances
        of nn.Module.

        Parameters:
            None

        Returns:
            None
        """
        num_classes = 1000  # Example number of classes
        crossvit_types = [
            "crossvit_tiny_240", "crossvit_small_240", "crossvit_base_240",
            "crossvit_9_240", "crossvit_15_240", "crossvit_18_240",
            "crossvit_9_dagger_240", "crossvit_15_dagger_240", "crossvit_15_dagger_408",
            "crossvit_18_dagger_240", "crossvit_18_dagger_408"
        ]

        for crossvit_type in crossvit_types:
            with self.subTest(crossvit_type=crossvit_type):
                model = get_crossvit_model(crossvit_type, num_classes)
                self.assertIsNotNone(model)
                self.assertIsInstance(model, nn.Module)
