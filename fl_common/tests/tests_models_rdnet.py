import os

from django.test import TestCase

from fl_common.models.rdnet import get_rdnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingRdnetTestCase(TestCase):
    """
    Test case class for processing RDnet models.
    """

    def test_known_rdnet_types(self):
        """
        Test the get_rdnet_model function with known RDNet types.

        For each known RDNet type, this test checks if the function returns a non-None RDNet model
        when provided with that type.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If the function returns None for any known RDNet type.
        """
        known_types = [
            "rdnet_tiny",
            "rdnet_small",
            "rdnet_base",
            "rdnet_large",
        ]

        for rdnet_type in known_types:
            with self.subTest(rdnet_type=rdnet_type):
                rdnet_model = get_rdnet_model(rdnet_type, num_classes=10)
                self.assertIsNotNone(rdnet_model)

    def test_unknown_rdnet_type(self):
        """
        Test the get_rdnet_model function behavior when provided with an unknown RDNet type.

        This test checks if the function raises a ValueError when it is called with an unknown RDNet type.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If the function does not raise a ValueError for an unknown RDNet type.
        """
        unknown_type = "unknown_rdnet_type"

        with self.assertRaises(ValueError):
            get_rdnet_model(unknown_type, num_classes=10)
