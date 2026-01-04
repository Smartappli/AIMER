import os

from django.test import TestCase

from fl_common.models.nextvit import get_nextvit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingNestTestCase(TestCase):
    """
    Test case class for processing Nextvit models.
    """

    def test_get_nextvit_model_known_types(self):
        """
        Test for known NEXTVIT architecture types to ensure they return a model without raising any exceptions.
        """
        known_nextvit_types = ["nextvit_small", "nextvit_base", "nextvit_large"]
        num_classes = 1000  # Assuming 1000 classes for the test

        for nextvit_type in known_nextvit_types:
            with self.subTest(nextvit_type=nextvit_type):
                try:
                    model = get_nextvit_model(nextvit_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{nextvit_type} should be a known NEXTVIT architecture.",
                    )

    def test_get_nextvit_model_unknown_type(self):
        """
        Test to ensure that an unknown NEXTVIT architecture type raises a ValueError.
        """
        unknown_nextvit_type = "unknown_nextvit_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_nextvit_model(unknown_nextvit_type, num_classes)
