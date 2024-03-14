import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.squeezenet import get_squeezenet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingSqueezenetestCase(TestCase):
    """
    Test case class for processing Squeezenet models.
    """

    def test_get_squeezenet_model(self):
        """
        Test case for obtaining various SqueezeNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of SqueezeNet model types to test
        squeezenet_types = ['SqueezeNet1_0', 'SqueezeNet1_1']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each SqueezeNet model type
        for squeezenet_type in squeezenet_types:
            with self.subTest(squeezenet_type=squeezenet_type):
                # Get the SqueezeNet model for testing
                model = get_squeezenet_model(squeezenet_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_squeezenet_unknown_architecture(self):
        """
        Test case for handling unknown SqueezeNet architecture in get_squeezenet_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown SqueezeNet architecture is provided.
        """
        squeezenet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a SqueezeNet model with an unknown architecture
            get_squeezenet_model(squeezenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown SqueezeNet Architecture: {squeezenet_type}'
        )

    def test_squeezenet_last_layer_adaptation(self):
        """
        Test case for ensuring last layer adaptation in SqueezeNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        squeezenet_type = 'SqueezeNet1_0'
        num_classes = 10

        # Override the last layer with a convolutional layer for testing purposes
        squeezenet_model = get_squeezenet_model(squeezenet_type, num_classes)
        last_layer = squeezenet_model.classifier[1]
        # Check if the last layer is an instance of nn.Conv2d
        self.assertIsInstance(last_layer, nn.Conv2d)
        self.assertEqual(last_layer.out_channels, num_classes)
        self.assertEqual(last_layer.kernel_size, (1, 1))
        self.assertEqual(last_layer.stride, (1, 1))
