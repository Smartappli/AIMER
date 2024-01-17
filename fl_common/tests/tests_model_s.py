import os
import torch.nn as nn
from django.test import TestCase
from fl_common.models.shufflenet import get_shufflenet_model
from fl_common.models.squeezenet import get_squeezenet_model
from fl_common.models.swin_transformer import get_swin_model

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"


class ProcessingPartSTestCase(TestCase):
    """Swim Model Unit Tests"""
    def test_get_swim_model(self):
        swin_types = ['Swin_T', 'Swin_S', 'Swin_B', 'Swin_V2_T', 'Swin_V2_S', 'Swin_V2_B']
        swin_types = ['Swin_T', 'Swin_S', 'Swin_B', 'Swin_V2_T', 'Swin_V2_S', 'Swin_V2_B']
        num_classes = 10  # You can adjust the number of classes as needed

        for swin_type in swin_types:
            with self.subTest(swin_type=swin_type):
                model = get_swin_model(swin_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_swin_unknown_architecture(self):
        swin_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_swin_model(swin_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown DenseNet Architecture : {swin_type}'
        )

    def test_swin_last_layer_adaptation(self):
        # Provide a known architecture type
        swin_type = 'Swin_T'
        num_classes = 10

        # Override the last layer with a linear layer for testing purposes
        swin_model = get_swin_model(swin_type, num_classes)
        last_layer = swin_model.head
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)

    def test_swin_model_structure(self):
        # Provide a known architecture type
        swin_type = 'Swin_T'
        num_classes = 10

        # Check if the model has a known structure with a linear last layer
        swin_model = get_swin_model(swin_type, num_classes)
        self.assertIsInstance(swin_model.head, nn.Linear)

    """SqueezeNet Model Unit Tests"""
    def test_get_squeezenet_model(self):
        squeezenet_types = ['SqueezeNet1_0', 'SqueezeNet1_1']
        num_classes = 10  # You can adjust the number of classes as needed

        for squeezenet_type in squeezenet_types:
            with self.subTest(squeezenet_type=squeezenet_type):
                model = get_squeezenet_model(squeezenet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_squeezenet_unknown_architecture(self):
        squeezenet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_squeezenet_model(squeezenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown SqueezeNet Architecture: {squeezenet_type}'
        )

    def test_squeezenet_last_layer_adaptation(self):
        # Provide a known architecture type
        squeezenet_type = 'SqueezeNet1_0'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        squeezenet_model = get_squeezenet_model(squeezenet_type, num_classes)
        last_layer = squeezenet_model.classifier[1]
        self.assertIsInstance(last_layer, nn.Conv2d)
        self.assertEqual(last_layer.out_channels, num_classes)
        self.assertEqual(last_layer.kernel_size, (1, 1))
        self.assertEqual(last_layer.stride, (1, 1))

    """ShuffleNet Model Unit Tests"""
    def test_get_shufflenet_model(self):
        shufflenet_types = ['ShuffleNet_V2_X0_5', 'ShuffleNet_V2_X1_0', 'ShuffleNet_V2_X1_5', 'ShuffleNet_V2_X2_0']
        num_classes = 10  # You can adjust the number of classes as needed

        for shufflenet_type in shufflenet_types:
            with self.subTest(shufflenet_type=shufflenet_type):
                model = get_shufflenet_model(shufflenet_type, num_classes)
                self.assertIsInstance(model, nn.Module)
                # Add more specific assertions about the model if needed

    def test_shufflenet_unknown_architecture(self):
        shufflenet_type = 'UnknownArchitecture'
        num_classes = 10

        with self.assertRaises(ValueError) as context:
            get_shufflenet_model(shufflenet_type, num_classes)

        self.assertEqual(
            str(context.exception),
            f'Unknown ShuffleNet Architecture: {shufflenet_type}'
        )

    def test_shufflenet_last_layer_adaptation(self):
        # Provide a known architecture type
        shufflenet_type = 'ShuffleNet_V2_X0_5'
        num_classes = 10

        # Override the last layer with a non-linear layer for testing purposes
        shufflenet_model = get_shufflenet_model(shufflenet_type, num_classes)
        last_layer = shufflenet_model.fc
        self.assertIsInstance(last_layer, nn.Linear)
        self.assertEqual(last_layer.out_features, num_classes)