import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.vision_transformer import get_vision_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingVisionTransformerTestCase(TestCase):
    """
    Test case class for processing Vision Transformer models.
    """

    def test_get_vision_model(self):
        """
        Test case for obtaining various Vision Transformer models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of Vision Transformer model types to test
        vision_types = ['ViT_B_16', 'ViT_B_32', 'ViT_L_16', 'ViT_L_32', 'ViT_H_14']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each Vision Transformer model type
        for vision_type in vision_types:
            with self.subTest(vision_type=vision_type):
                # Get the Vision Transformer model for testing
                model = get_vision_model(vision_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module, msg=f'get_vision_model {vision_type} KO')

    def test_vision_unknown_architecture(self):
        """
        Test case for handling unknown Vision Transformer architecture in get_vision_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown Vision Transformer architecture is provided.
        """
        vision_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a Vision Transformer model with an unknown architecture
            get_vision_model(vision_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown Vision Transformer Architecture: {vision_type}'
        )

    """
    def test_vision_nonlinear_last_layer(self):
        # Provide a vision_type with a known non-linear last layer
        vision_type = 'ViT_B_16'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        vision_model = get_vision_model(vision_type, num_classes)
        vision_model.heads[-1] = nn.ReLU()

        with self.assertRaises(ValueError) as context:
            # Try to create the vision model again
            get_vision_model(vision_type, num_classes)

        # Check if the raised ValueError contains the expected message
        self.assertIn(
            'The last layer is not a linear layer.',
            str(context.exception)
        )
    """
