import os

from django.test import TestCase

from fl_common.models.hiera import get_hiera_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingHieraTestCase(TestCase):
    """
    Test case class for processing Hiera models.
    """

    def test_known_hiera_types(self):
        """
        Test case to ensure that known HIERA architectures can be created.

        Iterates through a list of known HIERA architecture types and attempts to create models for each type.
        Verifies that the models are not None.

        Raises:
            AssertionError: If any known hiera architecture fails to be created or if any architecture is unknown.
        """
        known_hiera_types = [
            "hiera_tiny_224",
            "hiera_small_224",
            "hiera_base_224",
            "hiera_base_plus_224",
            "hiera_large_224",
            "hiera_huge_224",
        ]

        num_classes = 1000  # Assuming 1000 classes for the test

        for hiera_type in known_hiera_types:
            with self.subTest(hiera_type=hiera_type):
                try:
                    model = get_hiera_model(hiera_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{hiera_type} should be a known hiera architecture.",
                    )

    def test_unknown_hiera_type(self):
        """
        Test case to ensure that attempting to create a HIERA model with an unknown architecture type raises a ValueError.

        Verifies that a ValueError is raised when attempting to create a hiera model with an unknown architecture type.

        Raises:
            AssertionError: If creating a hiera model with an unknown architecture type does not raise a ValueError.
        """
        unknown_hiera_type = "unknown_hiera_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_hiera_model(unknown_hiera_type, num_classes)
