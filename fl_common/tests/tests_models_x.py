import os
import torch
from django.test import TestCase
from fl_common.models.xception import get_xception_model
from fl_common.models.xcit import get_xcit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartXTestCase(TestCase):
    """Xception Model Unit Tests"""

    def test_all_xception_models(self):
        """
        Test case for obtaining various Xception models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of Xception model types to test
        xception_types = [
            'legacy_xception',
            'xception41',
            'xception65',
            'xception71',
            'xception41p',
            'xception65p'
        ]

        # Loop through each Xception model type
        for xception_type in xception_types:
            with self.subTest(xception_type=xception_type):
                # Get the Xception model for testing
                model = get_xception_model(xception_type, num_classes=10)
                # Check if the model is an instance of torch.nn.Module
                self.assertTrue(isinstance(model, torch.nn.Module))

    def test_unknown_xception_type(self):
        """
        Test case for handling unknown Xception model type in get_xception_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown Xception model type is provided.
        """
        with self.assertRaises(ValueError):
            # Attempt to get an Xception model with an unknown type
            get_xception_model('unknown_type', num_classes=10)

    """XcIt Model Unit Tests"""
    def test_xcit_supported_models(self):
        """
        Test function for supported models in XCiT.

        Iterates through a list of supported XCiT models and checks if the model can be loaded successfully.

        Models:
        - xcit_nano_12_p16_224
        - xcit_nano_12_p16_384
        - xcit_tiny_12_p16_224
        - xcit_tiny_12_p16_384
        - xcit_small_12_p16_224
        - xcit_small_12_p16_384
        - xcit_tiny_24_p16_224
        - xcit_tiny_24_p16_384
        - xcit_small_24_p16_224
        - xcit_small_24_p16_384
        - xcit_medium_24_p16_224
        - xcit_medium_24_p16_384
        - xcit_large_24_p16_224
        - xcit_large_24_p16_384
        - xcit_nano_12_p8_224
        - xcit_nano_12_p8_384
        - xcit_tiny_12_p8_224
        - xcit_tiny_12_p8_384
        - xcit_small_12_p8_224
        - xcit_small_12_p8_384
        - xcit_tiny_24_p8_224
        - xcit_tiny_24_p8_384
        - xcit_small_24_p8_224
        - xcit_small_24_p8_384
        - xcit_medium_24_p8_224
        - xcit_medium_24_p8_384
        - xcit_large_24_p8_224
        - xcit_large_24_p8_384

        For each model, it attempts to load the model with the specified number of classes (10 in this case).
        """
        supported_models = [
            'xcit_nano_12_p16_224', 'xcit_nano_12_p16_384',
            'xcit_tiny_12_p16_224', 'xcit_tiny_12_p16_384',
            'xcit_small_12_p16_224', 'xcit_small_12_p16_384',
            'xcit_tiny_24_p16_224', 'xcit_tiny_24_p16_384',
            'xcit_small_24_p16_224', 'xcit_small_24_p16_384',
            'xcit_medium_24_p16_224', 'xcit_medium_24_p16_384',
            'xcit_large_24_p16_224', 'xcit_large_24_p16_384',
            'xcit_nano_12_p8_224', 'xcit_nano_12_p8_384',
            'xcit_tiny_12_p8_224', 'xcit_tiny_12_p8_384',
            'xcit_small_12_p8_224', 'xcit_small_12_p8_384',
            'xcit_tiny_24_p8_224', 'xcit_tiny_24_p8_384',
            'xcit_small_24_p8_224', 'xcit_small_24_p8_384',
            'xcit_medium_24_p8_224', 'xcit_medium_24_p8_384',
            'xcit_large_24_p8_224', 'xcit_large_24_p8_384'
        ]

        for model_type in supported_models:
            with self.subTest(model_type=model_type):
                model = get_xcit_model(model_type, num_classes=10)  # Specify the number of classes as needed
                self.assertIsNotNone(model)

    def test_unknown_xcit_model(self):
        """
        Test that attempting to create an XCiT model with an unknown model type raises a ValueError.
        """
        with self.assertRaises(ValueError):
            get_xcit_model('unknown_model', num_classes=10)

