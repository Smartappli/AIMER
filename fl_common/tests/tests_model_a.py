import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.alexnet import get_alexnet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartATestCase(TestCase):
    """AlexNet Model Unit Tests"""
    def test_alexnet_model(self):
        alexnet_types = ['AlexNet']
        num_classes = 10  # You can adjust the number of classes as needed

        for alexnet_type in alexnet_types:
            with self.subTest(alexnet_type=alexnet_type):
                model = get_alexnet_model(alexnet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed


    def test_alexnet_unknown_architecture(self):
        alexnet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_alexnet_model(alexnet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown AlexNet Architecture: {alexnet_type}'
        )


    def test_alexnet_last_layer_adaptation(self):
        # Provide a known architecture type
        alexnet_type = 'AlexNet'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        alexnet_model = get_alexnet_model(alexnet_type, num_classes)
        last_layer = alexnet_model.classifier[6]
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)