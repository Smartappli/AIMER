import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.densenet import get_densenet_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartCTestCase(TestCase):
    """DenseNet Model Unit Tests"""
    def test_densenet_model(self):
        # densenet = get_densenet_model('DenseNet121', 1000)
        # self.assertIsNotNone(densenet, msg="get_densenet_model KO")
        densenet_types = ['DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201']
        num_classes = 10  # You can adjust the number of classes as needed

        for densenet_type in densenet_types:
            with self.subTest(densenet_type=densenet_type):
                model = get_densenet_model(densenet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_densenet_unknown_architecture(self):
        densenet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_densenet_model(densenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown DenseNet Architecture : {densenet_type}'
        )

    def test_denseet_last_layer_adaptation(self):
        # Provide a known architecture type
        densenet_type = 'DenseNet121'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        densenet_model = get_densenet_model(densenet_type, num_classes)
        last_layer = densenet_model.classifier
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)