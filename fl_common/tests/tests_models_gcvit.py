import os
from django.test import TestCase
from fl_common.models.gcvit import get_gcvit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingGcVitTestCase(TestCase):
    """
    Test case class for processing Gcvit models.
    """

    def test_known_gcvit_types(self):
        """
        Test the get_gcvit_model function with known GCVIT types.

        For each known GCVIT type, this test checks if the function returns a non-None GCVIT model
        when provided with that type.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If the function returns None for any known GCVIT type.
        """
        known_types = [
            "gcvit_xxtiny",
            "gcvit_xtiny",
            "gcvit_tiny",
            "gcvit_small",
            "gcvit_base"]

        for gcvit_type in known_types:
            with self.subTest(gcvit_type=gcvit_type):
                gcvit_model = get_gcvit_model(gcvit_type, num_classes=10)
                self.assertIsNotNone(gcvit_model)

    def test_unknown_gcvit_type(self):
        """
        Test the get_gcvit_model function behavior when provided with an unknown GCVIT type.

        This test checks if the function raises a ValueError when it is called with an unknown GCVIT type.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If the function does not raise a ValueError for an unknown GCVIT type.
        """
        unknown_type = "unknown_gcvit_type"

        with self.assertRaises(ValueError):
            get_gcvit_model(unknown_type, num_classes=10)
