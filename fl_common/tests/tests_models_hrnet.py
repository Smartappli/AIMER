import os
from django.test import TestCase
from fl_common.models.hrnet import get_hrnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


# Hrnet models unit tests
class ProcessingHrnetTestCase(TestCase):
    """
    Test case class for processing Hrnet models.
    """

    def test_hrnet_model_known_types(self):
        """
        Test for known HRNet architecture types to ensure they return a model without raising any exceptions.
        """
        known_hrnet_types = [
            'hrnet_w18_small',
            'hrnet_w18_small_v2',
            'hrnet_w18',
            'hrnet_w30',
            'hrnet_w32',
            'hrnet_w40',
            'hrnet_w44',
            'hrnet_w48',
            'hrnet_w64',
            'hrnet_w18_ssld',
            'hrnet_w48_ssld'
        ]

        num_classes = 1000  # Assuming 1000 classes for the test

        for hrnet_type in known_hrnet_types:
            with self.subTest(hrnet_type=hrnet_type):
                try:
                    model = get_hrnet_model(hrnet_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(f"{hrnet_type} should be a known HRNet architecture.")

    def test_hrnet_model_unknown_type(self):
        """
        Test for an unknown HRNet architecture type to ensure it raises a ValueError.
        """
        unknown_hrnet_type = "unknown_hrnet_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_hrnet_model(unknown_hrnet_type, num_classes)
