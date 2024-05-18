import os
from django.test import TestCase
from fl_common.models.pnasnet import get_pnasnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPnasnetTestCase(TestCase):
    """
    Test case class for processing Pnasnet models.
    """

    def test_known_pnasnet_type(self):
        """
        Test for known PNASNet architecture types to ensure they return a model without raising any exceptions.
        """
        known_pnasnet_types = ["pnasnet5large"]
        num_classes = 1000  # Assuming 1000 classes for the test

        for pnasnet_type in known_pnasnet_types:
            with self.subTest(pnasnet_type=pnasnet_type):
                try:
                    model = get_pnasnet_model(pnasnet_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(f"{pnasnet_type} should be a known PNASNet architecture.")

    def test_unknown_pnasnet_type(self):
        """
        Test to ensure that an unknown PNASNet architecture type raises a ValueError.
        """
        unknown_pnasnet_type = "unknown_pnasnet_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_pnasnet_model(unknown_pnasnet_type, num_classes)
