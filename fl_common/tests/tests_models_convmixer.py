import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.convmixer import get_convmixer_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartCTestCase(TestCase):
     # Convmixer model unit tests
    def test_get_convmixer_model(self):
        """
        Unit test for the `get_convmixer_model` function.

        Iterates through different Convmixer architectures and verifies whether the function
        returns a valid model instance for each architecture type.

        Parameters:
            self: The test case object.

        Returns:
            None
        """
        num_classes = 1000  # Example number of classes
        convmixer_types = ["convmixer_1536_20", "convmixer_768_32", "convmixer_1024_20_ks9_p14"]

        for convmixer_type in convmixer_types:
            with self.subTest(convmixer_type=convmixer_type):
                model = get_convmixer_model(convmixer_type, num_classes)
                self.assertIsNotNone(model)
                self.assertIsInstance(model, nn.Module)
