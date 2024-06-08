import os

from django.test import TestCase

from fl_common.models.vitamin import get_vitamin_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingViTaminTestCase(TestCase):
    """
    Test case class for processing ViTamin models.
    """

    def test_known_vitamin_types(self):
        """
        Test the get_ViTamin_model function with known ViTamin types.

        For each known ViTamin type, this test checks if the function returns a non-None ViTamin model
        when provided with that type.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If the function returns None for any known ViTamin type.
        """
        known_types = [
            "vitamin_small_224",
            "vitamin_base_224",
            "vitamin_large_224",
            "vitamin_large_256",
            "vitamin_large_336",
            "vitamin_large_384",
            "vitamin_large2_224",
            "vitamin_large2_256",
            "vitamin_large2_336",
            "vitamin_large2_384",
            "vitamin_xlarge_256",
            "vitamin_xlarge_336",
            "vitamin_xlarge_384",
        ]

        for vitamin_type in known_types:
            with self.subTest(vitamin_type=vitamin_type):
                vitamin_model = get_vitamin_model(vitamin_type, num_classes=10)
                self.assertIsNotNone(vitamin_model)

    def test_unknown_vitamin_type(self):
        """
        Test the get_vitamin_model function behavior when provided with an unknown ViTamin type.

        This test checks if the function raises a ValueError when it is called with an unknown ViTamin type.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If the function does not raise a ValueError for an unknown ViTamin type.
        """
        unknown_type = "unknown_ViTamin_type"

        with self.assertRaises(ValueError):
            get_vitamin_model(unknown_type, num_classes=10)
