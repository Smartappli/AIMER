import os
import torch
from django.test import TestCase
from fl_common.models.inception_next import get_inception_next_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingInceptionNextTestCase(TestCase):
    """ "
    Test case class for processing Inception Next models.
    """

    def test_all_inception_next_models(self):
        """
        Test case for obtaining Inception-Next models with different architectures.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of Inception-Next model types to test
        inception_next_types = [
            "inception_next_tiny",
            "inception_next_small",
            "inception_next_base",
        ]

        # Loop through each Inception-Next model type
        for inception_next_type in inception_next_types:
            with self.subTest(inception_next_type=inception_next_type):
                # Get the Inception-Next model for testing
                model = get_inception_next_model(inception_next_type, num_classes=10)
                # Check if the model is an instance of torch.nn.Module
                self.assertTrue(isinstance(model, torch.nn.Module))

    def test_unknown_inception_next_type(self):
        """
        Test case for handling unknown Inception-Next architecture in get_inception_next_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown Inception-Next architecture is provided.
        """
        with self.assertRaises(ValueError):
            # Attempt to get an Inception-Next model with an unknown
            # architecture
            get_inception_next_model("unknown_type", num_classes=10)
