import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.gcvit import get_gcvit_model
from fl_common.models.googlenet import get_googlenet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartGTestCase(TestCase):
    """Gcvit Models Unit Tests"""
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
            "gcvit_xxtiny", "gcvit_xtiny", "gcvit_tiny",
            "gcvit_small", "gcvit_base"
        ]

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

    """GoogleNet Model Unit Tests"""
    def test_get_googlenet_model(self):
        """
        Test case for obtaining GoogleNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of GoogleNet model types to test
        googlenet_types = ['GoogLeNet']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each GoogleNet model type
        for googlenet_type in googlenet_types:
            with self.subTest(googlenet_type=googlenet_type):
                # Get the GoogleNet model for testing
                model = get_googlenet_model(googlenet_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_googlenet_unknown_architecture(self):
        """
        Test case for handling unknown GoogleNet architecture in get_googlenet_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown GoogleNet architecture is provided.
        """
        googlenet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a GoogleNet model with an unknown architecture
            get_googlenet_model(googlenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown AlexNet Architecture: {googlenet_type}'
        )

    def test_googlenet_last_layer_adaptation(self):
        """
        Test case for ensuring the last layer adaptation in GoogleNet models.

        Raises:
            AssertionError: If the assertion fails.
        """
        # Provide a known architecture type
        googlenet_type = 'GoogLeNet'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        googlenet_model = get_googlenet_model(googlenet_type, num_classes)
        last_layer = googlenet_model.fc
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)
