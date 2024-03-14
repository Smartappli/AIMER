import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.swin_transformer import get_swin_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingSwinTransformerTestCase(TestCase):
    """
    Test case class for processing Swin Transformer models.
    """

    def test_get_swin_model(self):
        """
        Test case for obtaining various Swin Transformer models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of Swin Transformer model types to test
        swin_types = ['Swin_T', 'Swin_S', 'Swin_B', 'Swin_V2_T', 'Swin_V2_S', 'Swin_V2_B']
        num_classes = 10  # You can adjust the number of classes as needed

        # Loop through each Swin Transformer model type
        for swin_type in swin_types:
            with self.subTest(swin_type=swin_type):
                # Get the Swin Transformer model for testing
                model = get_swin_model(swin_type, num_classes)
                # Check if the model is an instance of torch.nn.Module
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_swin_unknown_architecture(self):
        """
        Test case for handling unknown Swin Transformer architecture in get_swin_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown Swin Transformer architecture is provided.
        """
        swin_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            # Attempt to get a Swin Transformer model with an unknown architecture
            get_swin_model(swin_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown DenseNet Architecture : {swin_type}'
        )

    def test_swin_last_layer_adaptation(self):
        """
        Test case for ensuring last layer adaptation in Swin Transformer models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        swin_type = 'Swin_T'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        swin_model = get_swin_model(swin_type, num_classes)
        last_layer = swin_model.head
        # Check if the last layer is an instance of nn.Linear
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    def test_swin_model_structure(self):
        """
        Test case for ensuring the structure of Swin Transformer models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # Provide a known architecture type
        swin_type = 'Swin_T'
        num_classes = 10

        # Check if the model has a known structure with a linear last layer
        swin_model = get_swin_model(swin_type, num_classes)
        self.assertIsInstance(swin_model.head, nn.Linear)
