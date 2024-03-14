import os
import torch
import torch.nn as nn
from django.test import TestCase
from fl_common.models.inception import get_inception_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingInceptionTestCase(TestCase):
    """
    Test case class for processing Inception models.
    """

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
