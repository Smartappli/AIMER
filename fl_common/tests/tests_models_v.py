import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.vgg import get_vgg_model
from fl_common.models.vision_transformer import get_vision_model
from fl_common.models.volo import get_volo_model
from fl_common.models.vovnet import get_vovnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartVTestCase(TestCase):
    """Volo Model Unit Tests"""

    def test_all_volo_models(self):
        """
        Test case for obtaining various Volo models.

        Raises:
            AssertionError: If any of the assertions fail.
        """
        volo_types = ['volo_d1_224', 'volo_d1_384', 'volo_d2_224', 'volo_d2_384', 'volo_d3_224', 'volo_d3_448',
                      'volo_d4_224', 'volo_d4_448', 'volo_d5_224', 'volo_d5_448', 'volo_d5_512']

        for volo_type in volo_types:
            with self.subTest(volo_type=volo_type):
                # Get the Volo model for testing
                model = get_volo_model(volo_type, num_classes=10)
                # Check if the model is an instance of torch.nn.Module
                self.assertTrue(isinstance(model, nn.Module))

    def test_volo_unknown_type(self):
        """
        Test case for handling unknown Volo model type in get_volo_model function.

        Raises:
            AssertionError: If the assertion fails.
            ValueError: If an unknown Volo model type is provided.
        """
        with self.assertRaises(ValueError):
            # Attempt to get a Volo model with an unknown type
            get_volo_model('unknown_type', num_classes=10)

    """Vision Transformer Model Unit Tests"""

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

    """Vovnet Models Unit Tests"""
    def test_known_architecture(self):
        """
        Test the function with known architectures.

        For each known architecture, it checks that no exceptions are raised,
        and the returned model is not None.

        Supported architectures:
        - 'vovnet39a'
        - 'vovnet57a'
        - 'ese_vovnet19b_slim_dw'
        - 'ese_vovnet19b_dw'
        - 'ese_vovnet19b_slim'
        - 'ese_vovnet39b'
        - 'ese_vovnet57b'
        - 'ese_vovnet99b'
        - 'eca_vovnet39b'
        - 'ese_vovnet39b_evos'
        """
        architectures = [
            'vovnet39a',
            'vovnet57a',
            'ese_vovnet19b_slim_dw',
            'ese_vovnet19b_dw',
            'ese_vovnet19b_slim',
            'ese_vovnet39b',
            'ese_vovnet57b',
            'ese_vovnet99b',
            'eca_vovnet39b',
            'ese_vovnet39b_evos'
        ]

        for arch in architectures:
            with self.subTest(architecture=arch):
                # Ensure no exceptions are raised
                model = get_vovnet_model(arch, num_classes=10)
                self.assertIsNotNone(model)

    def test_unknown_architecture(self):
        """
        Test the function with known architectures.
        Ensure that no exceptions are raised, and the returned model is not None.
        """
        # Test the function with an unknown architecture
        with self.assertRaises(ValueError):
            get_vovnet_model('unknown_architecture', num_classes=10)
