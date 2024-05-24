import os
from django.test import TestCase
from fl_common.models.sknet import get_sknet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingSequencerTestCase(TestCase):
    """
    Test case class for processing Sknet models.
    """

    def test_known_sknet_types(self):
        """
        Test for known SKNet architecture types to ensure they return a model without raising any exceptions.
        """
        known_sknet_types = [
            "skresnet18",
            "skresnet34",
            "skresnet50",
            "skresnet50d",
            "skresnext50_32x4d",
        ]
        num_classes = 1000  # Assuming 1000 classes for the test

        for sknet_type in known_sknet_types:
            with self.subTest(sknet_type=sknet_type):
                try:
                    model = get_sknet_model(sknet_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{sknet_type} should be a known SKNet architecture."
                    )

    def test_unknown_sknet_type(self):
        """
        Test to ensure that an unknown SKNet architecture type raises a ValueError.
        """
        unknown_sknet_type = "unknown_sknet_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_sknet_model(unknown_sknet_type, num_classes)
