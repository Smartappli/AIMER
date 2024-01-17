import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.googlenet import get_googlenet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartGTestCase(TestCase):
    """GoogleNet Model Unit Tests"""
    def test_get_googlenet_model(self):
        # googlenet = get_googlenet_model('GoogLeNet', 1000)
        # self.assertIsNotNone(googlenet, msg="get_googlenet_model KO")
        googlenet_types = ['GoogLeNet']
        num_classes = 10  # You can adjust the number of classes as needed

        for googlenet_type in googlenet_types:
            with self.subTest(googlenet_type=googlenet_type):
                model = get_googlenet_model(googlenet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_googlenet_unknown_architecture(self):
        googlenet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_googlenet_model(googlenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown AlexNet Architecture: {googlenet_type}'
        )

    def test_googlenet_last_layer_adaptation(self):
        # Provide a known architecture type
        googlenet_type = 'GoogLeNet'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        googlenet_model = get_googlenet_model(googlenet_type, num_classes)
        last_layer = googlenet_model.fc
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)