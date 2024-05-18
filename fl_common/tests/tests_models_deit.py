import os
from django.test import TestCase
from fl_common.models.deit import get_deit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingDeitTestCase(TestCase):
    """
    Test case class for processing Deit models.
    """

    def test_known_deit_types(self):
        """
        Test case to ensure that known DEIT (Data-efficient Image Transformer) architectures can be created.

        Iterates through a list of known DEIT architecture types and attempts to create models for each type.
        Verifies that the models are not None.

        Raises:
            AssertionError: If any known DEIT architecture fails to be created or if any architecture is unknown.
        """
        known_deit_types = [
            "deit_tiny_patch16_224",
            "deit_small_patch16_224",
            "deit_base_patch16_224",
            "deit_base_patch16_384",
            "deit_tiny_distilled_patch16_224",
            "deit_small_distilled_patch16_224",
            "deit_base_distilled_patch16_224",
            "deit_base_distilled_patch16_384",
            "deit3_small_patch16_224",
            "deit3_small_patch16_384",
            "deit3_medium_patch16_224",
            "deit3_base_patch16_224",
            "deit3_base_patch16_384",
            "deit3_large_patch16_224",
            "deit3_large_patch16_384",
            "deit3_huge_patch14_224",
        ]

        num_classes = 1000  # Assuming 1000 classes for the test

        for deit_type in known_deit_types:
            with self.subTest(deit_type=deit_type):
                try:
                    model = get_deit_model(deit_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(f"{deit_type} should be a known Deit architecture.")

    def test_unknown_deit_type(self):
        """
        Test case to ensure that attempting to create a DEIT (Data-efficient Image Transformer) model with an unknown architecture type raises a ValueError.

        Verifies that a ValueError is raised when attempting to create a DEIT model with an unknown architecture type.

        Raises:
            AssertionError: If creating a DEIT model with an unknown architecture type does not raise a ValueError.
        """
        unknown_deit_type = "unknown_deit_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_deit_model(unknown_deit_type, num_classes)
