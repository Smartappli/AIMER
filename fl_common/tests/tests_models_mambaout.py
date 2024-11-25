import os

from django.test import TestCase
from torch import nn

from fl_common.models.mambaout import get_mambaout_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingMambaoutTestCase(TestCase):
    """
    Test case class for processing Mambaout models.
    """

    def test_valid_architecture(self):
        """Test for a valid Mambaout architecture."""
        architectures = [
            "mambaout_femto",
            "mambaout_kobe",
            "mambaout_tiny",
            "mambaout_small",
            "mambaout_base",
            "mambaout_small_rw",
            "mambaout_base_short_rw",
            "mambaout_base_tall_rw",
            "mambaout_base_wide_rw",
            "mambaout_base_plus_rw",
            "test_mambaout",
        ]
        num_classes = 1000  # Change this to the appropriate number of classes

        for mambaout_type in architectures:
            with self.subTest(mambaout_type=mambaout_type):
                result = get_mambaout_model(mambaout_type, num_classes)
                self.assertIsInstance(result, nn.Module)

    def test_unknown_architecture(self):
        """Test for an unknown Mambaout architecture."""
        mambaout_type = "unknown_architecture"
        num_classes = 1000  # Change this to the appropriate number of classes
        with self.assertRaises(ValueError):
            get_mambaout_model(mambaout_type, num_classes)

    def test_valid_architecture_custom_classes(self):
        """Test for a valid Mambaout architecture with a custom number of classes."""
        architectures = [
            "mambaout_femto",
            "mambaout_kobe",
            "mambaout_tiny",
            "mambaout_small",
            "mambaout_base",
            "mambaout_small_rw",
            "mambaout_base_short_rw",
            "mambaout_base_tall_rw",
            "mambaout_base_wide_rw",
            "mambaout_base_plus_rw",
            "test_mambaout",
        ]
        num_classes = 500  # Change this to a custom number of classes

        for mambaout_type in architectures:
            with self.subTest(mambaout_type=mambaout_type):
                result = get_mambaout_model(mambaout_type, num_classes)
                self.assertIsInstance(result, nn.Module)
