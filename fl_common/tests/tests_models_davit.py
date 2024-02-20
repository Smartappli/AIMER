import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.davit import get_davit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingDavitTestCase(TestCase):
    """Davit Models Unit Tests"""
    def test_all_davit_models(self):
        """
        Test the creation of all Davit models.

        Iterates through all valid Davit types and checks if the returned model is not None.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - AssertionError: If any of the Davit models is None.
        """
        davit_types = ['davit_tiny', 'davit_small', 'davit_base', 'davit_large', 'davit_huge', 'davit_giant']
        num_classes = 10

        for davit_type in davit_types:
            with self.subTest(davit_type=davit_type):
                davit_model = get_davit_model(davit_type, num_classes)
                self.assertIsNotNone(davit_model)

    def test_unknown_davit_type(self):
        """
        Test the behavior of get_davit_model when an unknown Davit type is specified.

        Parameters:
        - None

        Returns:
        - None

        Raises:
        - ValueError: If an unknown Davit architecture is specified.
        """
        with self.assertRaises(ValueError):
            get_davit_model('unknown_type', num_classes=10)
