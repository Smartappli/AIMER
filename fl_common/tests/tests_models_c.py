import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.convnext import get_convnext_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartCTestCase(TestCase):
    """ConvNeXt Model Unit Test"""
    def test_convnet_model(self):
        """
        Test case for creating ConvNeXt models.

        Iterates through different ConvNeXt architectures and checks if the created model is an instance
        of `nn.Module`.

        Raises:
            AssertionError: If the assertion fails.
        """
        convnext_types = ['ConvNeXt_Tiny', 'ConvNeXt_Small', 'ConvNeXt_Base', 'ConvNeXt_Large']
        num_classes = 10  # You can adjust the number of classes as needed

        for convnext_type in convnext_types:
            with self.subTest(convnext_type=convnext_type):
                model = get_convnext_model(convnext_type, num_classes)
                self.assertIsInstance(model, nn.Module)

    def test_convnext_unknown_architecture(self):
        """
        Test case for handling unknown ConvNeXt architecture.

        Raises:
            ValueError: If an unknown ConvNeXt architecture is encountered.
            AssertionError: If the assertion fails.
        """
        convnext_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_convnext_model(convnext_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown ConvNeXt Architecture : {convnext_type}'
        )

    def test_convnext_last_layer_adaptation(self):
        """
        Test case for ensuring the last layer adaptation in ConvNeXt models.

        Raises:
            AssertionError: If the assertion fails.
        """
        # Provide a known architecture type
        convnext_type = 'ConvNeXt_Large'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        convnext_model = get_convnext_model(convnext_type, num_classes)
        last_layer = None
        for layer in reversed(convnext_model.classifier):
            if isinstance(layer, nn.Linear):
                last_layer = layer
                break

        self.assertIsNotNone(last_layer)
        self.assertEqual(last_layer.out_features, num_classes)
