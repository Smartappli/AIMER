import os
import torch
import torch.nn as nn
from django.test import TestCase
from fl_common.models.inception_next import get_inception_next_model
from fl_common.models.inception import get_inception_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartITestCase(TestCase):
    """Inception Next Model Unit Tests"""

    def test_all_inception_next_models(self):
        """
        Test case for obtaining Inception-Next models with different architectures.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of Inception-Next model types to test
        inception_next_types = [
            'inception_next_tiny',
            'inception_next_small',
            'inception_next_base'
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
            # Attempt to get an Inception-Next model with an unknown architecture
            get_inception_next_model('unknown_type', num_classes=10)


    """Inception Model Unit Tests"""

    def test_get_inception_model(self):
        """
        Test case for obtaining Inception models with different architectures.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of Inception model types to test
        inception_types = [
            'Inception_V3',
            'inception_v4',
            'inception_resnet_v2'
        ]

        # Loop through each Inception model type
        for inception_type in inception_types:
            with self.subTest(inception_type=inception_type):
                # Get the Inception model for testing
                model = get_inception_model(inception_type, num_classes=10)
                # Check if the model is an instance of torch.nn.Module
                self.assertTrue(isinstance(model, torch.nn.Module))

    def test_inception_unknown_architecture(self):
        """
        Test case for handling unknown Inception architecture in get_inception_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown Inception architecture is provided.
        """
        inception_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get an Inception model with an unknown architecture
            get_inception_model(inception_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown Inception Architecture: {inception_type}'
        )

    def test_inception_last_layer_adaptation(self):
        """
        Test case for ensuring the last layer adaptation in Inception models.

        Raises:
            AssertionError: If the assertion fails.
        """
        # Provide a known architecture type
        inception_type = 'Inception_V3'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        inception_model = get_inception_model(inception_type, num_classes)
        last_layer = inception_model.fc
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)
