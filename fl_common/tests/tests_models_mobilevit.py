import os

from django.test import TestCase
from torch import nn

from fl_common.models.mobilevit import get_mobilevit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingMobilevitTestCase(TestCase):
    """
    Test case class for processing Mobilevit models.
    """

    def test_known_architecture(self):
        """
        Test if the function returns a valid model for a known MobileViT architecture type.
        """
        # Define a list of known MobileViT architecture types
        known_architectures = [
            "mobilevit_xxs",
            "mobilevit_xs",
            "mobilevit_s",
            "mobilevitv2_050",
            "mobilevitv2_075",
            "mobilevitv2_100",
            "mobilevitv2_125",
            "mobilevitv2_150",
            "mobilevitv2_175",
            "mobilevitv2_200",
        ]
        # Iterate over each known architecture type and test the function
        for architecture in known_architectures:
            with self.subTest(architecture=architecture):
                model = get_mobilevit_model(architecture, num_classes=1000)
                self.assertIsNotNone(model)
                self.assertIsInstance(model, nn.Module)

    def test_unknown_architecture(self):
        """
        Test if the function raises a ValueError for an unknown MobileViT architecture type.
        """
        # Test for an unknown MobileViT architecture type
        with self.assertRaises(ValueError):
            get_mobilevit_model("unknown_architecture", num_classes=1000)
