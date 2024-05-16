import os
from django.test import TestCase
from fl_common.models.xcit import get_xcit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartXTestCase(TestCase):
    """
    Test case class for processing Xcit models.
    """

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
            "xcit_nano_12_p16_224",
            "xcit_nano_12_p16_384",
            "xcit_tiny_12_p16_224",
            "xcit_tiny_12_p16_384",
            "xcit_small_12_p16_224",
            "xcit_small_12_p16_384",
            "xcit_tiny_24_p16_224",
            "xcit_tiny_24_p16_384",
            "xcit_small_24_p16_224",
            "xcit_small_24_p16_384",
            "xcit_medium_24_p16_224",
            "xcit_medium_24_p16_384",
            "xcit_large_24_p16_224",
            "xcit_large_24_p16_384",
            "xcit_nano_12_p8_224",
            "xcit_nano_12_p8_384",
            "xcit_tiny_12_p8_224",
            "xcit_tiny_12_p8_384",
            "xcit_small_12_p8_224",
            "xcit_small_12_p8_384",
            "xcit_tiny_24_p8_224",
            "xcit_tiny_24_p8_384",
            "xcit_small_24_p8_224",
            "xcit_small_24_p8_384",
            "xcit_medium_24_p8_224",
            "xcit_medium_24_p8_384",
            "xcit_large_24_p8_224",
            "xcit_large_24_p8_384",
        ]

        for model_type in supported_models:
            with self.subTest(model_type=model_type):
                # Specify the number of classes as needed
                model = get_xcit_model(model_type, num_classes=10)
                self.assertIsNotNone(model)

    def test_unknown_xcit_model(self):
        """
        Test that attempting to create an XCiT model with an unknown model type raises a ValueError.
        """
        with self.assertRaises(ValueError):
            get_xcit_model("unknown_model", num_classes=10)
