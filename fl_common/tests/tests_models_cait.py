import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.cait import get_cait_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingCaitTestCase(TestCase):
    """
    Test case class for processing Cait models.
    """

    def test_known_cait_types(self):
        """
        Test the get_cait_model function with known CAIT types.

        For each known CAIT type, this test checks if the function returns a non-None CAIT model
        when provided with that type.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If the function returns None for any known CAIT type.
        """
        known_types = [
            "cait_xxs24_224", "cait_xxs24_384", "cait_xxs36_224", "cait_xxs36_384",
            "cait_xs24_384", "cait_s24_224", "cait_s24_384", "cait_s36_384",
            "cait_m36_384", "cait_m48_448"
        ]

        for cait_type in known_types:
            with self.subTest(cait_type=cait_type):
                cait_model = get_cait_model(cait_type, num_classes=10)
                self.assertIsNotNone(cait_model)

    def test_unknown_cait_type(self):
        """
        Test the get_cait_model function behavior when provided with an unknown CAIT type.

        This test checks if the function raises a ValueError when it is called with an unknown CAIT type.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If the function does not raise a ValueError for an unknown CAIT type.
        """
        unknown_type = "unknown_cait_type"

        with self.assertRaises(ValueError):
            get_cait_model(unknown_type, num_classes=10)
