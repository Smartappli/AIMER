import os
from django.test import TestCase
from fl_common.models.levit import get_levit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartITestCase(TestCase):
    """
    Test case class for processing Levit models.
    """

    def test_known_levit_types(self):
        """
        Test the function with known Levit types.

        For each known Levit type, this test checks if the function returns a non-None Levit model
        when provided with that type.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If the function returns None for any known Levit type.
        """
        # Test the function with known Levit types
        known_types = [
            "levit_128s",
            "levit_128",
            "levit_192",
            "levit_256",
            "levit_384",
            "levit_384_s8",
            "levit_512_s8",
            "levit_512",
            "levit_256d",
            "levit_512d",
            "levit_conv_128s",
            "levit_conv_128",
            "levit_conv_192",
            "levit_conv_256",
            "levit_conv_384",
            "levit_conv_384_s8",
            "levit_conv_512_s8",
            "levit_conv_512",
            "levit_conv_256d",
            "levit_conv_512d",
        ]

        for levit_type in known_types:
            with self.subTest(levit_type=levit_type):
                levit_model = get_levit_model(levit_type, num_classes=10)
                self.assertIsNotNone(levit_model)

    def test_unknown_levit_type(self):
        """
        Test the function behavior when provided with an unknown Levit type.

        This test checks if the function raises a ValueError when it is called with an unknown Levit type.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If the function does not raise a ValueError for an unknown Levit type.
        """
        # Test the function with an unknown Levit type
        unknown_type = "unknown_levit_type"

        with self.assertRaises(ValueError):
            get_levit_model(unknown_type, num_classes=10)
