import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.maxvit import get_maxvit_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingMaxvitTestCase(TestCase):
    """
    Test case class for processing Maxvit models.
    """

    def test_get_maxvit_model(self):
        """
        Test case for obtaining MaxVit models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of MaxVit model types to test
        maxvit_types = ['MaxVit_T']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each MaxVit model type
        for maxvit_type in maxvit_types:
            with self.subTest(maxvit_type=maxvit_type):
                # Get the MaxVit model for testing
                model = get_maxvit_model(maxvit_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_maxvit_unknown_architecture(self):
        """
        Test case for handling unknown MaxVit architecture in get_maxvit_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown MaxVit architecture is provided.
        """
        maxvit_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a MaxVit model with an unknown architecture
            get_maxvit_model(maxvit_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown MaxVit Architecture: {maxvit_type}'
        )

    def test_maxvit_last_layer_adaptation(self):
        """
        Test case for ensuring last layer adaptation in MaxVit models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        maxvit_type = 'MaxVit_T'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        maxvit_model = get_maxvit_model(maxvit_type, num_classes)
        last_layer = maxvit_model.classifier[-1]
        # Check if the last layer is an instance of nn.Linear
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)
