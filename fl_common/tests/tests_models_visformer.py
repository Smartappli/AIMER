import os
from django.test import TestCase
from fl_common.models.visformer import get_visformer_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingVisformerTestCase(TestCase):
    """
    Test case class for processing Visformer models.
    """

    def test_known_visformer_types(self):
        """
        Test for known Visformer architecture types to ensure they return a model without raising any exceptions.
        """
        known_visformer_types = ['visformer_tiny', 'visformer_small']
        num_classes = 1000  # Assuming 1000 classes for the test

        for visformer_type in known_visformer_types:
            with self.subTest(visformer_type=visformer_type):
                try:
                    model = get_visformer_model(visformer_type, num_classes)
                    self.assertIsNotNone(model)
                except ValueError:
                    self.fail(
                        f"{visformer_type} should be a known Visformer architecture.")

    def test_unknown_visformer_type(self):
        """
        Test to ensure that an unknown Visformer architecture type raises a ValueError.
        """
        unknown_visformer_type = "unknown_visformer_type"
        num_classes = 1000

        with self.assertRaises(ValueError):
            get_visformer_model(unknown_visformer_type, num_classes)
