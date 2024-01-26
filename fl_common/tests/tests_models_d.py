import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.davit import get_davit_model
from fl_common.models.densenet import get_densenet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartCTestCase(TestCase):
    """Davit Models Unit Tests"""
    def test_all_davit_models(self):
        davit_types = ['davit_tiny', 'davit_small', 'davit_base', 'davit_large', 'davit_huge', 'davit_giant']
        num_classes = 10

        for davit_type in davit_types:
            with self.subTest(davit_type=davit_type):
                davit_model = get_davit_model(davit_type, num_classes)
                self.assertIsNotNone(davit_model)

    def test_unknown_davit_type(self):
        with self.assertRaises(ValueError):
            get_davit_model('unknown_type', num_classes=10)

    """DenseNet Models Unit Tests"""
    def test_densenet_model(self):
        """
        Test case for creating DenseNet models.

        Iterates through different DenseNet architectures and checks if the created model is an instance
        of `nn.Module`.

        Raises:
            AssertionError: If the assertion fails.
        """
        # List of DenseNet architectures to test
        densenet_types = ['DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201']
        num_classes = 10  # You can adjust the number of classes as needed

        for densenet_type in densenet_types:
            with self.subTest(densenet_type=densenet_type):
                model = get_densenet_model(densenet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_densenet_unknown_architecture(self):
        """
        Test case for handling unknown DenseNet architecture.

        Raises:
            ValueError: If an unknown DenseNet architecture is encountered.
            AssertionError: If the assertion fails.
        """
        densenet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_densenet_model(densenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown DenseNet Architecture : {densenet_type}'
        )

    def test_densenet_last_layer_adaptation(self):
        """
        Test case for ensuring the last layer adaptation in DenseNet models.

        Raises:
            AssertionError: If the assertion fails.
        """
        # Provide a known architecture type
        densenet_type = 'DenseNet121'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        densenet_model = get_densenet_model(densenet_type, num_classes)
        last_layer = densenet_model.classifier
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)
