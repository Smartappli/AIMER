import os
from django.test import TestCase
from fl_common.models.hardcorenas import get_hardcorenas_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingHardcorenasTestCase(TestCase):
    # Hardcorenas model unit tests
    def test_known_hardcorenas_types(self):
        """
        Test for known HardcoreNAS architecture types to ensure they return a model without raising any exceptions.
        """
        known_hardcorenas_types = [
            'hardcorenas_a',
            'hardcorenas_b',
            'hardcorenas_c',
            'hardcorenas_d',
            'hardcorenas_e',
            'hardcorenas_f'
        ]

        num_classes = 1000  # Assuming 1000 classes for the test

        for hardcorenas_type in known_hardcorenas_types:
            with self.subTest(hardcorenas_type=hardcorenas_type):
                try:
                    model = get_hardcorenas_model(hardcorenas_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(f"{hardcorenas_type} should be a known HardcoreNAS architecture.")

    def test_unknown_hardcorenas_type(self):
        """
        Test for an unknown HardcoreNAS architecture type to ensure it raises a ValueError.
        """
        unknown_hardcorenas_type = "unknown_hardcorenas_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_hardcorenas_model(unknown_hardcorenas_type, num_classes)
