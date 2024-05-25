import os
from django.test import TestCase
from fl_common.models.twins import get_twins_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingTwinsTestCase(TestCase):
    """
    Test case class for processing Twins models.
    """

    def test_known_twins_types(self):
        """
        Test for known Twins architecture types to ensure they return a model without raising any exceptions.
        """
        known_twins_types = [
            "twins_pcpvt_small",
            "twins_pcpvt_base",
            "twins_pcpvt_large",
            "twins_svt_small",
            "twins_svt_base",
            "twins_svt_large",
        ]

        num_classes = 1000  # Assuming 1000 classes for the test

        for twins_type in known_twins_types:
            with self.subTest(twins_type=twins_type):
                try:
                    model = get_twins_model(twins_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{twins_type} should be a known Twins architecture.",
                    )

    def test_unknown_twins_type(self):
        """
        Test for an unknown Twins architecture type to ensure it raises a ValueError.
        """
        unknown_twins_type = "unknown_twins_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_twins_model(unknown_twins_type, num_classes)
