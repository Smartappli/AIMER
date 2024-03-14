import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.regnet import get_regnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingRegnetTestCase(TestCase):
    """
    Test case class for processing Regnet models.
    """

    def test_get_regnet_model(self):
        """
        Test case for obtaining various RegNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of RegNet model types to test
        regnet_types = ['RegNet_X_400MF', 'RegNet_X_800MF', 'RegNet_X_1_6GF', 'RegNet_X_3_2GF', 'RegNet_X_16GF',
                        'RegNet_Y_400MF', 'RegNet_Y_800MF', 'RegNet_Y_1_6GF', 'RegNet_Y_3_2GF', 'RegNet_Y_16GF']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each RegNet model type
        for regnet_type in regnet_types:
            with self.subTest(regnet_type=regnet_type):
                # Get the RegNet model for testing
                model = get_regnet_model(regnet_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_regnet_unknown_architecture(self):
        """
        Test case for handling unknown RegNet architecture in get_regnet_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown RegNet architecture is provided.
        """
        regnet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a RegNet model with an unknown architecture
            get_regnet_model(regnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown RegNet Architecture: {regnet_type}'
        )

    def test_regnet_last_layer_adaptation(self):
        """
        Test case for ensuring last layer adaptation in RegNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        regnet_type = 'RegNet_X_400MF'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        regnet_model = get_regnet_model(regnet_type, num_classes)
        last_layer = regnet_model.fc
        # Check if the last layer is an instance of nn.Linear
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)
