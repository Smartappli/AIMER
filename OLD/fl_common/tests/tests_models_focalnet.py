import os

from django.test import TestCase

from fl_common.models.focalnet import get_focalnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingFastVitTestCase(TestCase):
    """
    Test case class for processing Focalnet models.
    """

    def test_known_focalnet_types(self):
        """
        Test case to ensure that known FocalNet architectures can be created successfully.

        Iterates through a list of known FocalNet architecture types and attempts to create models for each type.
        Verifies that the models are not None.

        Raises:
            AssertionError: If any known FocalNet architecture fails to be created.
        """
        known_focalnet_types = [
            "focalnet_tiny_srf",
            "focalnet_small_srf",
            "focalnet_base_srf",
            "focalnet_tiny_lrf",
            "focalnet_small_lrf",
            "focalnet_base_lrf",
            "focalnet_large_fl3",
            "focalnet_large_fl4",
            "focalnet_xlarge_fl3",
            "focalnet_xlarge_fl4",
            "focalnet_huge_fl3",
            "focalnet_huge_fl4",
        ]
        num_classes = 1000  # Example number of classes

        for focalnet_type in known_focalnet_types:
            with self.subTest(focalnet_type=focalnet_type):
                try:
                    model = get_focalnet_model(focalnet_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{focalnet_type} should be a known Focalnet architecture.",
                    )

    def test_unknown_focalnet_type(self):
        """
        Test case to ensure that attempting to create a FocalNet model with an unknown architecture type raises a
        ValueError.

        Verifies that a ValueError is raised when attempting to create a FocalNet model with an unknown architecture
        type.

        Raises:
            AssertionError: If creating a FocalNet model with an unknown architecture type does not raise a ValueError.
        """
        unknown_focalnet_type = "unknown_focalnet_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_focalnet_model(unknown_focalnet_type, num_classes)
