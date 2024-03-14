import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.vgg import get_vgg_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingVGGTestCase(TestCase):
    """
    Test case class for processing VGG models.
    """
    def test_get_vgg_model(self):
        """
        Test case for obtaining various VGG models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of VGG model types to test
        vgg_types = ['VGG11', 'VGG11_BN', 'VGG13', 'VGG13_BN', 'VGG16', 'VGG16_BN', 'VGG19', 'VGG19_BN']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each VGG model type
        for vgg_type in vgg_types:
            with self.subTest(vgg_type=vgg_type):
                # Get the VGG model for testing
                model = get_vgg_model(vgg_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_vgg_unknown_architecture(self):
        """
        Test case for handling unknown VGG architecture in get_vgg_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown VGG architecture is provided.
        """
        vgg_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a VGG model with an unknown architecture
            get_vgg_model(vgg_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown VGG Architecture : {vgg_type}'
        )

    def test_vgg_last_layer_adaptation(self):
        """
        Test case for ensuring last layer adaptation in VGG models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        vgg_type = 'VGG16'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        vgg_model = get_vgg_model(vgg_type, num_classes)
        last_layer = vgg_model.classifier[-1]
        # Check if the last layer is an instance of nn.Linear
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    def test_vgg_model_structure(self):
        """
        Test case for ensuring the structure of VGG models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        vgg_type = 'VGG16'
        num_classes = 10

        # Check if the model has a known structure with a linear last layer
        vgg_model = get_vgg_model(vgg_type, num_classes)
        self.assertIsInstance(vgg_model.classifier[-1], nn.Linear)
