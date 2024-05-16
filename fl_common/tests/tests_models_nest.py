import os
from django.test import TestCase
from fl_common.models.nest import get_nest_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingNestTestCase(TestCase):
    """
    Test case class for processing Nest models.
    """

    def test_known_nest_types(self):
        """
        Test for known Nest architecture types to ensure they return a model without raising any exceptions.
        """
        known_nest_types = [
            "nest_base",
            "nest_small",
            "nest_tiny",
            "nest_base_jx",
            "nest_small_jx",
            "nest_tiny_jx",
        ]
        num_classes = 1000  # Assuming 1000 classes for the test

        for nest_type in known_nest_types:
            with self.subTest(nest_type=nest_type):
                try:
                    model = get_nest_model(nest_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{nest_type} should be a known Nest architecture.")

    def test_unknown_nest_type(self):
        """
        Test to ensure that an unknown Nest architecture type raises a ValueError.
        """
        unknown_nest_type = "unknown_nest_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_nest_model(unknown_nest_type, num_classes)
