import os
from django.test import TestCase
from fl_common.models.senet import get_senet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingSenetTestCase(TestCase):
    """
    Test case class for processing Senet models.
    """

    def test_known_types(self):
        """
        Test for known SENet architecture types to ensure they return a model without raising any exceptions.
        """
        known_types = [
            "legacy_seresnet18",
            "legacy_seresnet34",
            "legacy_seresnet50",
            "legacy_seresnet101",
            "legacy_seresnet152",
            "legacy_senet154",
            "legacy_seresnext26_32x4d",
            "legacy_seresnext50_32x4d",
            "legacy_seresnext101_32x4d",
        ]
        num_classes = 1000  # Assuming 1000 classes for the test

        for senet_type in known_types:
            with self.subTest(senet_type=senet_type):
                try:
                    model = get_senet_model(senet_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{senet_type} should be a known SENet architecture."
                    )

    def test_unknown_type(self):
        """
        Test to ensure that an unknown SENet architecture type raises a ValueError.
        """
        unknown_type = "unknown_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_senet_model(unknown_type, num_classes)
