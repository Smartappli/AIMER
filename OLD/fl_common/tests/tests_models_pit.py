import os

from django.test import TestCase

from fl_common.models.pit import get_pit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPitTestCase(TestCase):
    """
    Test case class for processing Pit models.
    """

    def test_known_pit_types(self):
        """
        Test for known PIT architecture types to ensure they return a model without raising any exceptions.
        """
        known_pit_types = [
            "pit_b_224",
            "pit_s_224",
            "pit_xs_224",
            "pit_ti_224",
            "pit_b_distilled_224",
            "pit_s_distilled_224",
            "pit_xs_distilled_224",
            "pit_ti_distilled_224",
        ]
        num_classes = 1000  # Assuming 1000 classes for the test

        for pit_type in known_pit_types:
            with self.subTest(pit_type=pit_type):
                try:
                    model = get_pit_model(pit_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(f"{pit_type} should be a known PIT architecture.")

    def test_unknown_pit_type(self):
        """
        Test to ensure that an unknown PIT architecture type raises a ValueError.
        """
        unknown_pit_type = "unknown_pit_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_pit_model(unknown_pit_type, num_classes)
