import os

from django.test import TestCase

from fl_common.models.eva import get_eva_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingEvaTestCase(TestCase):
    """
    Test case class for processing Eva models.
    """

    def test_known_eva_types(self):
        """
        Test for known Eva architecture types to ensure they return a model without raising any exceptions.
        """
        known_eva_types = [
            "vit_medium_patch16_rope_reg1_gap_256",
            "vit_mediumd_patch16_rope_reg1_gap_256",
            "vit_betwixt_patch16_rope_reg4_gap_256",
            "vit_base_patch16_rope_reg1_gap_256",
        ]

        num_classes = 1000  # Assuming 1000 classes for the test

        for eva_type in known_eva_types:
            with self.subTest(eva_type=eva_type):
                try:
                    model = get_eva_model(eva_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(f"{eva_type} should be a known Eva architecture.")

    def test_unknown_eva_type(self):
        """
        Test for an unknown Eva architecture type to ensure it raises a ValueError.
        """
        unknown_eva_type = "unknown_eva_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_eva_model(unknown_eva_type, num_classes)
