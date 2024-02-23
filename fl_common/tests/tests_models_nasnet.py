import os
from django.test import TestCase
from fl_common.models.nasnet import get_nasnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingNasnetTestCase(TestCase):
    """Nasnet Model Unit Tests"""

    def test_known_nasnet_type(self):
        """
        Test for known NASNet architecture types to ensure they return a model without raising any exceptions.
        """
        known_nasnet_types = ['nasnetalarge']
        num_classes = 1000  # Assuming 1000 classes for the test

        for nasnet_type in known_nasnet_types:
            with self.subTest(nasnet_type=nasnet_type):
                try:
                    model = get_nasnet_model(nasnet_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(f"{nasnet_type} should be a known NASNet architecture.")

    def test_unknown_nasnet_type(self):
        """
        Test to ensure that an unknown NASNet architecture type raises a ValueError.
        """
        unknown_nasnet_type = "unknown_nasnet_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_nasnet_model(unknown_nasnet_type, num_classes)
