import os
import torch
from django.test import TestCase
from fl_common.models.xception import get_xception_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingXceptionTestCase(TestCase):
    """
    Test case class for processing Xception models.
    """

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
