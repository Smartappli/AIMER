import os

from django.test import TestCase

from fl_common.models.selecsls import get_selecsls_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingSelecslsTestCase(TestCase):
    """
    Test case class for processing Selecsls models.
    """

    def test_known_selecsls_types(self):
        """
        Test for known SelecSLS architecture types to ensure they return a model without raising any exceptions.
        """
        known_selecsls_types = [
            "selecsls42",
            "selecsls42b",
            "selecsls60",
            "selecsls60b",
            "selecsls84",
        ]
        num_classes = 1000  # Assuming 1000 classes for the test

        for selecsls_type in known_selecsls_types:
            with self.subTest(selecsls_type=selecsls_type):
                try:
                    model = get_selecsls_model(selecsls_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{selecsls_type} should be a known SelecSLS architecture.",
                    )

    def test_unknown_selecsls_type(self):
        """
        Test to ensure that an unknown SelecSLS architecture type raises a ValueError.
        """
        unknown_selecsls_type = "unknown_selecsls_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_selecsls_model(unknown_selecsls_type, num_classes)
