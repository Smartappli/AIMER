import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.shufflenet import get_shufflenet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartSTestCase(TestCase):
    """
    Test case class for processing Shufflenet models.
    """

    def test_get_shufflenet_model(self):
        """
        Test case for obtaining various ShuffleNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of ShuffleNet model types to test
        shufflenet_types = [
            'ShuffleNet_V2_X0_5',
            'ShuffleNet_V2_X1_0',
            'ShuffleNet_V2_X1_5',
            'ShuffleNet_V2_X2_0']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each ShuffleNet model type
        for shufflenet_type in shufflenet_types:
            with self.subTest(shufflenet_type=shufflenet_type):
                # Get the ShuffleNet model for testing
                model = get_shufflenet_model(shufflenet_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_shufflenet_unknown_architecture(self):
        """
        Test case for handling unknown ShuffleNet architecture in get_shufflenet_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown ShuffleNet architecture is provided.
        """
        shufflenet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a ShuffleNet model with an unknown architecture
            get_shufflenet_model(shufflenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown ShuffleNet Architecture: {shufflenet_type}'
        )

    def test_shufflenet_last_layer_adaptation(self):
        """
        Test case for ensuring last layer adaptation in ShuffleNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        shufflenet_type = 'ShuffleNet_V2_X0_5'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        shufflenet_model = get_shufflenet_model(shufflenet_type, num_classes)
        last_layer = shufflenet_model.fc
        # Check if the last layer is an instance of nn.Linear
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)
