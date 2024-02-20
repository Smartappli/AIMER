import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.resnet import get_resnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingResnetTestCase(TestCase):
    """ResNet Mmodel Unit Tests"""
    def test_get_resnet_model(self):
        """
        Test case for obtaining various ResNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of ResNet model types to test
        resnet_types = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each ResNet model type
        for resnet_type in resnet_types:
            with self.subTest(resnet_type=resnet_type):
                # Get the ResNet model for testing
                model = get_resnet_model(resnet_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_resnet_unknown_architecture(self):
        """
        Test case for handling unknown ResNet architecture in get_resnet_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown ResNet architecture is provided.
        """
        resnet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a ResNet model with an unknown architecture
            get_resnet_model(resnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown ResNet Architecture: {resnet_type}'
        )

    def test_resnet_last_layer_adaptation(self):
        """
        Test case for ensuring last layer adaptation in ResNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        resnet_type = 'ResNet18'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        resnet_model = get_resnet_model(resnet_type, num_classes)
        last_layer = resnet_model.fc
        # Check if the last layer is an instance of nn.Linear
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)
