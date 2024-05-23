import os
from torch import nn
from django.test import TestCase
from fl_common.models.convit import get_convit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingConvitTestCase(TestCase):
    """
    Test case class for processing Canvit models.
    """

    def test_valid_architecture(self):
        """Test for a valid Convit architecture."""
        architectures = [
            "convit_tiny",
            "convit_small",
            "convit_base",
        ]
        num_classes = 1000  # Change this to the appropriate number of classes

        for convit_type in architectures:
            with self.subTest(convit_type=convit_type):
                result = get_convit_model(convit_type, num_classes)
                self.assertIsInstance(result, nn.Module)

    def test_unknown_architecture(self):
        """Test for an unknown Convit architecture."""
        convit_type = "unknown_architecture"
        num_classes = 1000  # Change this to the appropriate number of classes
        with self.assertRaises(ValueError):
            get_convit_model(convit_type, num_classes)

    def test_valid_architecture_custom_classes(self):
        """Test for a valid Convit architecture with a custom number of classes."""
        architectures = [
            "convit_tiny",
            "convit_small",
            "convit_base",
        ]
        num_classes = 500  # Change this to a custom number of classes

        for convit_type in architectures:
            with self.subTest(convit_type=convit_type):
                result = get_convit_model(convit_type, num_classes)
                self.assertIsInstance(result, nn.Module)
