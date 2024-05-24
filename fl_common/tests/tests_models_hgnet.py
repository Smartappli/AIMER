import os
from django.test import TestCase
from fl_common.models.hgnet import get_hgnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingHgnetTestCase(TestCase):
    """
    Test case class for processing Hgnet models.
    """

    def test_known_hgnet_types(self):
        """
        Test for known Hgnet architecture types to ensure they return a model without raising any exceptions.
        """
        known_hgnet_types = [
            "hgnet_tiny",
            "hgnet_small",
            "hgnet_base",
            "hgnetv2_b0",
            "hgnetv2_b1",
            "hgnetv2_b2",
            "hgnetv2_b3",
            "hgnetv2_b4",
            "hgnetv2_b5",
            "hgnetv2_b6",
        ]

        num_classes = 1000  # Assuming 1000 classes for the test

        for hgnet_type in known_hgnet_types:
            with self.subTest(hgnet_type=hgnet_type):
                try:
                    model = get_hgnet_model(hgnet_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{hgnet_type} should be a known Hgnet architecture."
                    )

    def test_unknown_hgnet_type(self):
        """
        Test for an unknown Hgnet architecture type to ensure it raises a ValueError.
        """
        unknown_hgnet_type = "unknown_hgnet_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_hgnet_model(unknown_hgnet_type, num_classes)
