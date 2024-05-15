import os
from django.test import TestCase
from fl_common.models.coat import get_coat_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingCoatTestCase(TestCase):
    """
    Test case class for processing Coat models.
    """

    def test_known_coat_types(self):
        """
        Test the get_coat_model function with known COAT types.

        For each known COAT type, this test checks if the function returns a non-None COAT model
        when provided with that type.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If the function returns None for any known COAT type.
        """
        known_types = [
            "coat_tiny", "coat_mini", "coat_small",
            "coat_lite_tiny", "coat_lite_mini", "coat_lite_small",
            "coat_lite_medium", "coat_lite_medium_384"
        ]

        for coat_type in known_types:
            with self.subTest(coat_type=coat_type):
                coat_model = get_coat_model(coat_type, num_classes=10)
                self.assertIsNotNone(coat_model)

    def test_unknown_coat_type(self):
        """
        Test the get_coat_model function behavior when provided with an unknown COAT type.

        This test checks if the function raises a ValueError when it is called with an unknown COAT type.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If the function does not raise a ValueError for an unknown COAT type.
        """
        unknown_type = "unknown_coat_type"

        with self.assertRaises(ValueError):
            get_coat_model(unknown_type, num_classes=10)
