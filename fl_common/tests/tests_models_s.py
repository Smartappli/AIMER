import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.shufflenet import get_shufflenet_model
from fl_common.models.squeezenet import get_squeezenet_model
from fl_common.models.swin_transformer import get_swin_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartSTestCase(TestCase):
    """Swim Model Unit Tests"""
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

    """SqueezeNet Model Unit Tests"""
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

    """ShuffleNet Model Unit Tests"""
    def test_get_shufflenet_model(self):
        """
        Test case for obtaining various ShuffleNet models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        # List of ShuffleNet model types to test
        shufflenet_types = ['ShuffleNet_V2_X0_5', 'ShuffleNet_V2_X1_0', 'ShuffleNet_V2_X1_5', 'ShuffleNet_V2_X2_0']
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
