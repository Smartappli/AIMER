import os
from django.test import TestCase
from fl_common.models.tresnet import get_tresnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingTresnetTestCase(TestCase):
    """
    Test case class for processing Tresnet models.
    """

    def test_known_tresnet_types(self):
        """
        Test for known TResNet architecture types to ensure they return a model without raising any exceptions.
        """
        known_tresnet_types = [
            "tresnet_m",
            "tresnet_l",
            "tresnet_xl",
            "tresnet_v2_l"]

        num_classes = 1000  # Assuming 1000 classes for the test

        for tresnet_type in known_tresnet_types:
            with self.subTest(tresnet_type=tresnet_type):
                try:
                    model = get_tresnet_model(tresnet_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{tresnet_type} should be a known TResNet architecture.")

    def test_unknown_tresnet_type(self):
        """
        Test for an unknown TResNet architecture type to ensure it raises a ValueError.
        """
        unknown_tresnet_type = "unknown_tresnet_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_tresnet_model(unknown_tresnet_type, num_classes)
