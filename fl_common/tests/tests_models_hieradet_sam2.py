import os

from django.test import TestCase

from fl_common.models.hieradet_sam2 import get_hieradet_sam2_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingHieradetSAM2TestCase(TestCase):
    """
    Test case class for processing Hieradet SAM2 models.
    """

    def test_known_hieradet_sam2_types(self):
        """
        Test case to ensure that known HIERADET SAM2 architectures can be created.

        Iterates through a list of known HIERADET SAM2 architecture types and attempts to create models for each type.
        Verifies that the models are not None.

        Raises:
            AssertionError: If any known hiera architecture fails to be created or if any architecture is unknown.
        """
        known_hiera_types = [
            "sam2_hiera_tiny",
            "sam2_hiera_small",
            "sam2_hiera_base_plus",
            "sam2_hiera_large",
            "hieradet_small",
        ]

        num_classes = 1000  # Assuming 1000 classes for the test

        for hieradet_sam2_type in known_hieradet_sam2_types:
            with self.subTest(hieradet_sam2_type=hieradet_sam2_type):
                try:
                    model = get_hieradet_sam2_model(
                        hieradet_sam2_type, num_classes
                    )
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{hieradet_sam2_type} should be a known hieradet sam2 architecture.",
                    )

    def test_unknown_hieradet_sam2_type(self):
        """
        Test case to ensure that attempting to create a HIERADET SAM2 model with an unknown architecture type raises a ValueError.

        Verifies that a ValueError is raised when attempting to create a hiera model with an unknown architecture type.

        Raises:
            AssertionError: If creating a HIERADET SAM2 model with an unknown architecture type does not raise a ValueError.
        """
        unknown_hiera_type = "unknown_hieradet_sam2_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_hieradet_sam2_model(unknown_hieradet_sam2_type, num_classes)
