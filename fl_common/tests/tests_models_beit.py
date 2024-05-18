import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.beit import get_beit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingBeitTestCase(TestCase):
    """
    Test case class for processing Beit models.
    """

    def test_valid_architecture(self):
        """Test for a valid BEiT architecture."""
        architectures = [
            "beit_base_patch16_224",
            "beit_base_patch16_384",
            "beit_large_patch16_224",
            "beit_large_patch16_384",
            "beit_large_patch16_512",
            "beitv2_base_patch16_224",
            "beitv2_large_patch16_224",
        ]
        num_classes = 1000  # Change this to the appropriate number of classes

        for beit_type in architectures:
            with self.subTest(beit_type=beit_type):
                result = get_beit_model(beit_type, num_classes)
                self.assertIsInstance(result, nn.Module)

    def test_unknown_architecture(self):
        """Test for an unknown BEiT architecture."""
        beit_type = "unknown_architecture"
        num_classes = 1000  # Change this to the appropriate number of classes
        with self.assertRaises(ValueError):
            get_beit_model(beit_type, num_classes)

    def test_valid_architecture_custom_classes(self):
        """Test for a valid BEiT architecture with a custom number of classes."""
        architectures = [
            "beit_base_patch16_224",
            "beit_base_patch16_384",
            "beit_large_patch16_224",
            "beit_large_patch16_384",
            "beit_large_patch16_512",
            "beitv2_base_patch16_224",
            "beitv2_large_patch16_224",
        ]
        num_classes = 500  # Change this to a custom number of classes

        for beit_type in architectures:
            with self.subTest(beit_type=beit_type):
                result = get_beit_model(beit_type, num_classes)
                self.assertIsInstance(result, nn.Module)
