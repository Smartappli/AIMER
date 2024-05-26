import os

from django.test import TestCase

from fl_common.models.tnt import get_tnt_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingTntTestCase(TestCase):
    """
    Test case class for processing Tnt models.
    """

    def test_known_tnt_types(self):
        """
        Test for known TnT architecture types to ensure they return a model without raising any exceptions.
        """
        known_tnt_types = ["tnt_s_patch16_224", "tnt_b_patch16_224"]
        num_classes = 1000  # Assuming 1000 classes for the test

        for tnt_type in known_tnt_types:
            with self.subTest(tnt_type=tnt_type):
                try:
                    model = get_tnt_model(tnt_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(f"{tnt_type} should be a known TnT architecture.")

    def test_unknown_tnt_type(self):
        """
        Test to ensure that an unknown TnT architecture type raises a ValueError.
        """
        unknown_tnt_type = "unknown_tnt_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_tnt_model(unknown_tnt_type, num_classes)
